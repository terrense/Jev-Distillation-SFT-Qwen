"""Explicit, budgeted candidate generation via a user-configured chat-completions teacher."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from intent_lab import ROOT, POLICY, request_json, validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=["compat", "deployed"], default="compat")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--container", default="hgdoctor-diagnostic-service-1")
    p.add_argument("--api-key-env", default="TEACHER_API_KEY")
    p.add_argument("--shard", required=True, help="unique owner/scenario batch id")
    p.add_argument("--coverage", required=True)
    p.add_argument("--batches", type=int, default=1)
    p.add_argument("--per-batch", type=int, default=20)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.provider == "compat" and (not a.model or not a.base_url):
        raise ValueError("compat requires --base-url and --model")
    if a.provider == "compat" and not a.base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
        raise ValueError("use HTTPS or localhost")
    if not 1 <= a.per_batch <= 50 or not 1 <= a.batches <= 1000:
        raise ValueError("invalid explicit generation budget")
    prompt = (ROOT / "config/generation_prompt.txt").read_text(encoding="utf-8")
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("choose a new output file for each shard; avoid silent duplicate generation")
    with path.open("w", encoding="utf-8") as f:
        for batch in range(a.batches):
            batch_id = f"{a.shard}-b{batch:04}"
            payload = {
                "model": a.model, "messages": [{"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"policy": POLICY, "batch_id": batch_id,
                 "coverage": a.coverage, "count": a.per_batch}, ensure_ascii=False)}],
                "temperature": .85, "max_tokens": 12000,
                "response_format": {"type": "json_object"}}
            if a.provider == "deployed":
                worker = '''import json,sys,urllib.request
from shared.utils.chat_api import _get_model_config
c=_get_model_config();p=json.load(sys.stdin);p['model']=c['model_name'];p['chat_template_kwargs']={'enable_thinking':False}
h={'Content-Type':'application/json'}
if c.get('api_key'):h['Authorization']='Bearer '+c['api_key']
r=urllib.request.Request(c['base_url'].rstrip('/')+'/chat/completions',data=json.dumps(p,ensure_ascii=False).encode(),headers=h)
with urllib.request.urlopen(r,timeout=150) as x: print(x.read().decode())
'''
                run = subprocess.run(["docker", "exec", "-i", a.container, "python", "-c", worker],
                    input=json.dumps(payload, ensure_ascii=False), capture_output=True, text=True, encoding="utf-8", timeout=165)
                if run.returncode:
                    raise RuntimeError("deployed generation failed; no candidate labels were approved")
                response = json.loads(run.stdout)
            else:
                response = request_json(a.base_url.rstrip("/") + "/chat/completions", payload, os.environ.get(a.api_key_env, ""), 150)
            raw_content = response["choices"][0]["message"]["content"]
            # Preserve failed batches for diagnosis; they still never enter accepted data.
            raw_folder = path.parent / "quarantine_raw"
            raw_folder.mkdir(exist_ok=True)
            (raw_folder / (batch_id + ".json")).write_text(raw_content, encoding="utf-8")
            candidates = json.loads(raw_content)["samples"]
            # Ignore claimed reviewer status. A generating model cannot approve its own labels.
            for i, row in enumerate(candidates):
                row.update(id=f"{batch_id}-{i:03}", source="llm_generated", review_status="pending",
                           policy_version=POLICY["version"], generator_model=response.get("model", a.model), shard=a.shard)
                row["family_id"] = batch_id + ":" + str(row["family_id"])
            validate(candidates)
            for row in candidates:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({"batch": batch_id, "count": len(candidates), "usage": response.get("usage")}), flush=True)


if __name__ == "__main__":
    main()
