"""Standard-library CLI for annotation, leakage checks, batch prediction and evaluation.

Never exports patient logs. All remote calls are explicit and request-budgeted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import shutil
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "config/policy.json").read_text(encoding="utf-8"))
LABELS = list(POLICY["labels"])
OLD_MAP = {"triage": "triage", "initial_consultation": "preconsult", "info_inquiry": "info",
           "registration": "info", "idle": "other"}


def system_prompt():
    return ("你是医院入口意图分类器。\n" + "\n".join(f"{k}: {v}" for k, v in POLICY["labels"].items())
            + "\n" + "\n".join(POLICY["rules"]))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def state_for(row):
    return {"context": row.get("context", {"stage": "unknown", "active_task": None}),
            "history": row.get("history", [])[-4:], "user_input": row["user_input"]}


def norm(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def state_hash(row):
    return hashlib.sha256(json.dumps(state_for(row), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate(rows):
    seen = set()
    for row in rows:
        for key in ("id", "family_id", "user_input", "source", "review_status"):
            if not row.get(key):
                raise ValueError(f"missing {key}: {row.get('id')}")
        if row["id"] in seen:
            raise ValueError(f"duplicate id: {row['id']}")
        seen.add(row["id"])
        if row.get("label") is not None and row["label"] not in LABELS:
            raise ValueError(f"invalid label: {row['id']}")
        if row.get("policy_version") != POLICY["version"]:
            raise ValueError(f"policy mismatch: {row['id']}")
        ctx = row.get("context", {})
        if set(ctx) - {"stage", "active_task"}:
            raise ValueError(f"unexpected context fields: {row['id']}")
        if ctx.get("stage", "unknown") not in {"unknown", "unregistered", "registered", "preconsulting", "after_visit"}:
            raise ValueError(f"invalid stage: {row['id']}")
        if ctx.get("active_task") not in {None, *LABELS}:
            raise ValueError(f"invalid active_task: {row['id']}")
        if len(row.get("history", [])) > 4:
            raise ValueError(f"history exceeds four entries: {row['id']}")
        for entry in row.get("history", []):
            if not isinstance(entry, dict) or entry.get("role") not in {"user", "assistant"} or not isinstance(entry.get("content"), str) or not entry["content"].strip():
                raise ValueError(f"history must contain role/content objects: {row['id']}")
        if not isinstance(row["user_input"], str) or not row["user_input"].strip() or '\ufffd' in row["user_input"]:
            raise ValueError(f"invalid text encoding or empty input: {row['id']}")


def request_json(url, payload, key="", timeout=60, transport="urllib"):
    if transport == "powershell":
        # Canonical public API via the same system HTTP client used in the smoke probe.
        # Secret is passed over stdin, never in the child command line or an on-disk file.
        executable = shutil.which("pwsh")
        if not executable:
            raise ValueError("PowerShell 7 (pwsh) is required for this transport")
        script = r'''
[Console]::InputEncoding=[Text.Encoding]::UTF8
[Console]::OutputEncoding=[Text.Encoding]::UTF8
$x=[Console]::In.ReadToEnd() | ConvertFrom-Json -AsHashtable
$h=@{}
if ($x.key) {$h.Authorization='Bearer '+$x.key}
$b=ConvertTo-Json -InputObject $x.payload -Depth 40 -Compress
try {
 $r=Invoke-WebRequest -Method Post -Uri $x.url -Headers $h -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($b)) -MaximumRedirection 0 -TimeoutSec $x.timeout
 [Console]::Out.Write($r.Content)
} catch {
 $status=[int]$_.Exception.Response.StatusCode
 [Console]::Error.Write('HTTP_'+$status)
 exit 1
} finally {$h.Clear();$x.key=$null}
'''
        r = subprocess.run([executable, "-NoProfile", "-NonInteractive", "-Command", script],
                           input=json.dumps({"url": url, "payload": payload, "key": key, "timeout": timeout}, ensure_ascii=False),
                           capture_output=True, text=True, encoding="utf-8", timeout=timeout+15)
        if r.returncode:
            raise RuntimeError(r.stderr.strip() if re.fullmatch(r"HTTP_\d+", r.stderr.strip()) else "transport_failed")
        return json.loads(r.stdout.lstrip("\ufeff"))
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Keep transport errors distinct from classification; do not retry billing/auth failures.
        raise RuntimeError(f"HTTP_{exc.code}") from None


def extract_label(content):
    raw = str(content or "").strip()
    if raw in LABELS:
        return raw, True
    # Log a format failure even if a recoverable JSON label is present.
    try:
        value = json.loads(raw).get("intent")
        return (value if value in LABELS else None), False
    except (ValueError, AttributeError):
        return None, False


def predict_remote(row, provider, args):
    state = state_for(row)
    key = os.environ.get(args.api_key_env, "")
    if provider == "jev":
        if not key:
            raise ValueError(f"missing environment variable: {args.api_key_env}")
        # Fixed canonical endpoint; never discover an auth endpoint by guessing domains.
        result = request_json("https://jevstation.com/api/v1/systemone", {
            "model": args.model or "jev", "state": state,
            "questions": {"intent": {"type": "choice", "instructions": "\n".join(POLICY["rules"]),
                                     "criteria": POLICY["labels"]}}}, key, args.timeout, args.transport)
        data = result.get("data", result)
        answer = data["answers"]["intent"]
        label = answer.get("choice")
        if label not in LABELS:
            raise ValueError("unexpected Jev label")
        return {"prediction": label, "format_valid": True, "answer": answer, "teacher_model": data.get("model", args.model or "jev"),
                "usage": data.get("usage", result.get("usage")), "credits": data.get("credits")}
    if not args.base_url or not args.model:
        raise ValueError("compat requires --base-url and --model")
    if not args.base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
        raise ValueError("compat endpoint must be HTTPS or localhost; use deployed docker mode for existing intranet service")
    payload = {"model": args.model, "messages": [{"role": "system", "content": system_prompt()},
               {"role": "user", "content": json.dumps(state, ensure_ascii=False)}],
               "temperature": 0, "max_tokens": 24}
    if args.qwen:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    result = request_json(args.base_url.rstrip("/") + "/chat/completions", payload, key, args.timeout, args.transport)
    raw = result["choices"][0]["message"].get("content")
    label, valid = extract_label(raw)
    return {"prediction": label, "format_valid": valid, "raw_output": raw, "usage": result.get("usage"), "teacher_model": args.model}


def docker_predictions(rows, args):
    """Only classification calls; no dispatch, registration or other HIS actions."""
    worker = r'''
import json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
x=json.load(sys.stdin)
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**kw): return None
def one(row):
 start=time.perf_counter()
 result={'id':row['id'],'state_hash':row['state_hash']}
 try:
  headers={'Content-Type':'application/json'}
  if x['provider']=='deployed-original':
   url='http://127.0.0.1:8000/api/v1/intent/recognize'
   body={'user_input':row['user_input'],'session_id':'intent-lab-'+row['id'],
         'context':{'round':0,'history':row.get('history',[])}}
  else:
   from shared.utils.chat_api import _get_model_config
   c=_get_model_config()
   result['teacher_model']=c['model_name']
   url=c['base_url'].rstrip('/')+'/chat/completions'
   if c.get('api_key'): headers['Authorization']='Bearer '+c['api_key']
   state={'context':row.get('context',{}),'history':row.get('history',[])[-4:],'user_input':row['user_input']}
   body={'model':c['model_name'],'messages':[{'role':'system','content':x['prompt']},
         {'role':'user','content':json.dumps(state,ensure_ascii=False)}],
         'temperature':0,'max_tokens':24,'chat_template_kwargs':{'enable_thinking':False}}
  req=urllib.request.Request(url,data=json.dumps(body,ensure_ascii=False).encode(),headers=headers)
  with urllib.request.build_opener(NoRedirect()).open(req,timeout=x['timeout']) as r: d=json.load(r)
  if x['provider']=='deployed-original':
   raw=d.get('intent'); result.update(raw_intent=raw,prediction=x['old_map'].get(raw),format_valid=raw in x['old_map'])
   result['clarification_returned']=bool(d.get('context'))
  else:
   raw=d['choices'][0]['message'].get('content','').strip()
   result.update(prediction=raw if raw in x['labels'] else None,format_valid=raw in x['labels'],raw_output=raw,usage=d.get('usage'))
   result['response_model']=d.get('model')
 except urllib.error.HTTPError as e: result['error']='HTTP_'+str(e.code)
 except Exception as e: result['error']=type(e).__name__
 result['latency_ms']=round((time.perf_counter()-start)*1000,2)
 return result
with ThreadPoolExecutor(max_workers=x['workers']) as pool:
 for r in pool.map(one,x['rows']): print(json.dumps(r,ensure_ascii=False),flush=True)
'''
    data = {"provider": args.provider, "prompt": system_prompt(), "labels": LABELS,
            "old_map": OLD_MAP, "workers": args.workers, "timeout": args.timeout,
            "rows": [{**r, "state_hash": state_hash(r)} for r in rows]}
    proc = subprocess.Popen(["docker", "exec", "-i", args.container, "python", "-c", worker],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    proc.stdin.write(json.dumps(data, ensure_ascii=False))
    proc.stdin.close()
    for line in proc.stdout:
        yield json.loads(line)
    errors = proc.stderr.read()
    if proc.wait():
        raise RuntimeError("docker worker failed: " + errors[-500:])


def predict(args):
    rows = read_jsonl(args.input)
    validate(rows)
    identity = {"provider": args.provider, "model": args.model, "base_url": args.base_url,
                "policy": POLICY, "qwen": args.qwen, "input": str(Path(args.input).resolve())}
    signature = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    meta = Path(str(args.output) + ".meta.json")
    if Path(args.output).exists():
        if not meta.exists() or json.loads(meta.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("cache identity changed: choose a new output file")
        previous = read_jsonl(args.output)
    else:
        previous = []
    cache = {(r["id"], r["state_hash"]) for r in previous if not r.get("error")}
    pending = [r for r in rows if (r["id"], state_hash(r)) not in cache][:args.max_requests]
    save_json(meta, {"signature": signature, **identity, "max_requests_this_run": args.max_requests,
                     "structured_context_supported_by_original": False})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    if args.provider.startswith("deployed-"):
        iterator = docker_predictions(pending, args)
    else:
        def remote_iter():
            def one(row):
                start = time.perf_counter()
                result = {"id": row["id"], "state_hash": state_hash(row)}
                try:
                    result.update(predict_remote(row, args.provider, args))
                except Exception as exc:
                    result["error"] = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
                result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
                return result
            source = iter(pending)
            stopped, consecutive_errors = False, 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                active = set()
                def add_one():
                    row = next(source, None)
                    if row is not None:
                        active.add(pool.submit(one, row))
                for _ in range(args.workers):
                    add_one()
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        active.remove(future)
                        result = future.result()
                        consecutive_errors = consecutive_errors + 1 if result.get("error") else 0
                        if consecutive_errors >= 3 or result.get("error") in {"HTTP_402", "HTTP_401", "HTTP_403"}:
                            stopped = True
                        yield result
                    if not stopped:
                        while len(active) < args.workers:
                            before = len(active)
                            add_one()
                            if len(active) == before:
                                break
        iterator = remote_iter()
    count = 0
    with Path(args.output).open("a", encoding="utf-8") as file:
        for result in iterator:
            result["provider"] = args.provider
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()
            count += 1
            errors = errors + 1 if result.get("error") else 0
            print(json.dumps({"completed": count, "id": result["id"], "prediction": result.get("prediction"),
                              "error": result.get("error")}, ensure_ascii=False), flush=True)
            # Remote iterator stops scheduling on auth/balance failure, then records all in-flight results.


def percentile(values, q):
    if not values:
        return None
    a = sorted(values)
    return a[max(0, math.ceil(q * len(a)) - 1)]


def score(args):
    gold = read_jsonl(args.gold)
    validate(gold)
    if any(r.get("review_status") != "human_approved" for r in gold) and not args.allow_provisional:
        raise ValueError("gold needs human_approved labels; smoke only: --allow-provisional")
    predictions = {r["id"]: r for r in read_jsonl(args.predictions)}
    confusion = {y: {p: 0 for p in LABELS + ["ERROR"]} for y in LABELS}
    failures = []
    latencies = []
    tag_stats = defaultdict(lambda: [0, 0])
    correct = 0
    format_ok = 0
    for row in gold:
        pred = predictions.get(row["id"], {})
        if pred and pred.get("state_hash") != state_hash(row):
            raise ValueError(f"stale prediction: {row['id']}")
        p = pred.get("prediction")
        valid = not pred.get("error") and p in LABELS and bool(pred.get("format_valid"))
        p = p if valid else "ERROR"
        confusion[row["label"]][p] += 1
        ok = row["label"] == p
        correct += ok
        format_ok += valid
        for tag in row.get("tags", []):
            tag_stats[tag][0] += ok
            tag_stats[tag][1] += 1
        if "latency_ms" in pred:
            latencies.append(pred["latency_ms"])
        if not ok:
            failures.append({"id": row["id"], "expected": row["label"], "actual": p,
                             "user_input": row["user_input"], "error": pred.get("error"),
                             "raw_output": pred.get("raw_output"), "raw_intent": pred.get("raw_intent")})
    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[y][label] for y in LABELS)
        precision = tp / predicted if predicted else 0
        recall = tp / actual if actual else 0
        per_label[label] = {"support": actual, "precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0}
    in_scope = sum(sum(confusion[y].values()) for y in LABELS if y != "other")
    reject_den = sum(confusion["other"].values())
    result = {"n": len(gold), "provisional": args.allow_provisional,
              "accuracy": correct / len(gold), "macro_f1": statistics.mean(x["f1"] for x in per_label.values()),
              "macro_f1_supported_labels": statistics.mean(x["f1"] for x in per_label.values() if x["support"] > 0),
              "missing_gold_labels": [label for label in LABELS if not per_label[label]["support"]],
              "format_success_rate": format_ok / len(gold), "per_label": per_label, "confusion": confusion,
              "false_rejection_rate": sum(confusion[y]["other"] for y in LABELS if y != "other") / in_scope if in_scope else None,
              "out_of_scope_acceptance_rate": sum(confusion["other"][y] for y in LABELS if y != "other") / reject_den if reject_den else None,
              "latency_p50_ms": percentile(latencies, .5), "latency_p95_ms": percentile(latencies, .95),
              "coverage": {t: {"correct": v[0], "n": v[1], "accuracy": v[0] / v[1]} for t, v in tag_stats.items()},
              "failures": failures,
              "warning": "Assistant-authored smoke labels are not an independent human benchmark." if args.allow_provisional else None}
    save_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"failures", "confusion"}}, ensure_ascii=False, indent=2))


def split(args):
    rows = read_jsonl(args.input)
    validate(rows)
    approved = [r for r in rows if r.get("label") in LABELS and r["review_status"] in {"human_approved", "teacher_consensus"}]
    if not approved:
        raise ValueError("no approved rows; pending candidates must not silently become training labels")
    # Group by explicit scenario family, real session/person, and exact normalized text.
    # This intentionally keeps repeated short utterances in one split, even across contexts.
    parent = {r["id"]: r["id"] for r in approved}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owners = {}
    for row in approved:
        keys = [("family", row["family_id"]), ("text", norm(row["user_input"]))]
        keys += [(k, row[k]) for k in ("session_group", "person_group", "scenario_group") if row.get(k)]
        for key in keys:
            if key in owners:
                parent[find(row["id"])] = find(owners[key])
            else:
                owners[key] = row["id"]
    groups = defaultdict(list)
    for r in approved:
        groups[find(r["id"])].append(r)
    out = {"train": [], "dev": [], "audit_holdout": []}
    for key, group in sorted(groups.items()):
        frozen = {r["split"] for r in group if r.get("split")}
        if len(frozen) > 1 or frozen - set(out):
            raise ValueError("merged family crosses frozen splits; quarantine instead of reshuffling")
        stable_key = min(r["family_id"] for r in group)
        if frozen:
            part = next(iter(frozen))
        elif args.seed_only:
            part = "train"
        else:
            value = int(hashlib.sha256((str(args.seed) + stable_key).encode()).hexdigest()[:8], 16) / 2**32
            part = "train" if value < .8 else "dev" if value < .9 else "audit_holdout"
        out[part].extend({**r, "split": part} for r in group)
    for name, values in out.items():
        save_jsonl(Path(args.output) / (name + ".jsonl"), values)
    save_json(Path(args.output) / "manifest.json", {
        "counts": {k: len(v) for k, v in out.items()}, "input": len(rows), "approved": len(approved),
        "policy_version": POLICY["version"], "seed": args.seed,
        "note": "audit_holdout is NOT the independent human blind test. Semantic clustering still required."})
    print(json.dumps({k: len(v) for k, v in out.items()}))


def export_sft(args):
    rows = read_jsonl(args.input)
    validate(rows)
    out = []
    for row in rows:
        if row["review_status"] not in {"human_approved", "teacher_consensus"}:
            raise ValueError(f"unapproved: {row['id']}")
        out.append({"prompt": [{"role": "system", "content": system_prompt()},
                              {"role": "user", "content": json.dumps(state_for(row), ensure_ascii=False)}],
                    "completion": [{"role": "assistant", "content": row["label"]}]})
    save_jsonl(args.output, out)
    save_json(str(args.output) + ".manifest.json", {"n": len(out), "policy_version": POLICY["version"],
              "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
              "policy_sha256": hashlib.sha256(json.dumps(POLICY, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
    print(json.dumps({"exported": len(out)}))


def consensus(args):
    rows = read_jsonl(args.input)
    validate(rows)
    a = {x["id"]: x for x in read_jsonl(args.teacher_a)}
    b = {x["id"]: x for x in read_jsonl(args.teacher_b)}
    accepted, review = [], []
    for row in rows:
        x, y = a.get(row["id"], {}), b.get(row["id"], {})
        matching_state = x.get("state_hash") == y.get("state_hash") == state_hash(row)
        agree = matching_state and not x.get("error") and not y.get("error") and x.get("format_valid") and y.get("format_valid") and x.get("prediction") in LABELS and x.get("prediction") == y.get("prediction")
        if x.get("teacher_model") and x.get("teacher_model") == y.get("teacher_model"):
            raise ValueError("two calls to the same model are not independent consensus")
        jev_answers = [t.get("answer", {}).get("probabilities", {}) for t in (x, y) if t.get("provider") == "jev"]
        for probabilities in jev_answers:
            ranked = sorted(probabilities.values(), reverse=True)
            if len(ranked) < 2 or ranked[0] < args.jev_min_prob or ranked[0]-ranked[1] < args.jev_min_margin:
                agree = False
        item = {**row, "teacher_a": x, "teacher_b": y}
        if row.get("review_status") == "human_approved":
            accepted.append(item)  # Human label takes precedence over teacher agreement.
        elif agree and not set(row.get("tags", [])) & {"boundary", "emergency", "injection", "mixed_intent", "needs_human"}:
            item.update(label=x["prediction"], review_status="teacher_consensus")
            accepted.append(item)
        else:
            item["review_status"] = "pending"
            review.append(item)
    save_jsonl(Path(args.output) / "accepted.jsonl", accepted)
    save_jsonl(Path(args.output) / "human_review_queue.jsonl", review)
    print(json.dumps({"accepted": len(accepted), "needs_review": len(review)}))


def audit(args):
    files = [Path(p) for p in args.files]
    rows = {str(p): read_jsonl(p) for p in files}
    for data in rows.values():
        validate(data)
    findings = []
    owners = {}
    for path, data in rows.items():
        for row in data:
            keys = [("family", row["family_id"]), ("state", state_hash(row)), ("normalized_text", norm(row["user_input"]))]
            keys += [(key, row[key]) for key in ("session_group", "person_group", "scenario_group") if row.get(key)]
            for key in keys:
                if key in owners and owners[key][0] != path:
                    findings.append({"kind": key[0], "left": owners[key], "right": [path, row["id"]]})
                owners[key] = (path, row["id"])
    save_json(args.output, {"cross_split_leaks": findings, "counts": {p: len(r) for p, r in rows.items()},
                          "limitation": "Exact/group check only. Embedding/MinHash semantic-cluster audit is still required."})
    print(json.dumps({"cross_split_leaks": len(findings)}))
    if findings:
        raise SystemExit(2)


def main():
    global POLICY, LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(ROOT / "config/policy.json"), help="use the frozen policy for historical experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--provider", choices=["jev", "compat", "deployed-original", "deployed-short"], required=True)
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env", default="TEACHER_API_KEY")
    p.add_argument("--qwen", action="store_true")
    p.add_argument("--container", default="hgdoctor-diagnostic-service-1")
    p.add_argument("--max-requests", type=int, default=10)
    p.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=1)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--transport", choices=["urllib", "powershell"], default="urllib")
    p.set_defaults(func=predict)
    p = sub.add_parser("score")
    p.add_argument("--gold", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-provisional", action="store_true")
    p.set_defaults(func=score)
    p = sub.add_parser("split")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261118)
    p.add_argument("--seed-only", action="store_true", help="approved seed data goes only to train")
    p.set_defaults(func=split)
    p = sub.add_parser("export-sft")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=export_sft)
    p = sub.add_parser("consensus")
    p.add_argument("--input", required=True)
    p.add_argument("--teacher-a", required=True)
    p.add_argument("--teacher-b", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--jev-min-prob", type=float, default=.9)
    p.add_argument("--jev-min-margin", type=float, default=.3)
    p.set_defaults(func=consensus)
    p = sub.add_parser("audit")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=audit)
    args = parser.parse_args()
    POLICY = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    LABELS = list(POLICY["labels"])
    if hasattr(args, "max_requests") and args.max_requests < 1:
        parser.error("--max-requests must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
