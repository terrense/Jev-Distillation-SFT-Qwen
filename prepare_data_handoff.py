"""Build pending review packets; never converts teacher agreement into human gold."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from intent_lab import ROOT, POLICY, LABELS, read_jsonl, save_json, save_jsonl, state_hash, validate


def main():
    dest = ROOT / "data/handoff_v0.2"
    rows = read_jsonl(ROOT / "data/seed_candidates.jsonl") + read_jsonl(ROOT / "data/context_pairs_v0.2.jsonl")
    validate(rows)
    old = {state_hash(r): r for r in read_jsonl(ROOT / "data/pilot_v0.1_candidates.jsonl")}
    teacher_files = [("jev", ROOT / "reports/jev_seed.jsonl", "hospital-intent-v0.1-draft"),
                     ("local_teacher", ROOT / "reports/second_teacher_pilot_v0.1.jsonl", "hospital-intent-v0.1-draft"),
                     ("jev", ROOT / "reports/jev_context_v0.2.jsonl", POLICY["version"])]
    teachers = {}
    for name, path, version in teacher_files:
        if path.exists():
            for r in read_jsonl(path):
                if not r.get("error"):
                    teachers.setdefault(r["state_hash"], {})[name] = {**r, "teacher_policy_version": version}
    handoff, priority, blind, mapping = [], [], [], []
    for row in rows:
        annotations = teachers.get(state_hash(row), {})
        reasons = []
        if "jev" not in annotations:
            reasons.append("no_jev_annotation")
        values = [v.get("prediction") for v in annotations.values()]
        if any(v != row.get("label") for v in values) or len(set(values)) > 1:
            reasons.append("teacher_disagreement")
        if "boundary" in row.get("tags", []):
            reasons.append("boundary_requires_review")
        if any(v["teacher_policy_version"] != POLICY["version"] for v in annotations.values()):
            reasons.append("historical_policy_requires_review")
        item = {**row, "proposed_label": row.get("label"), "teacher_annotations": annotations,
                "review_status": "pending", "review_reasons": reasons,
                "reviewer_a_label": None, "reviewer_b_label": None, "adjudicated_label": None, "review_comment": ""}
        handoff.append(item)
        if reasons:
            priority.append(item)
        opaque = "sample-" + hashlib.sha256(row["id"].encode()).hexdigest()[:12]
        blind.append({"id": opaque, "family_id": "group-" + hashlib.sha256(row["family_id"].encode()).hexdigest()[:12],
                      "user_input": row["user_input"], "history": row.get("history", []), "context": row.get("context", {}),
                      "source": "review_packet", "policy_version": POLICY["version"], "review_status": "pending",
                      "label": None, "action_if_unclear": None, "comment": ""})
        mapping.append({"packet_id": opaque, "original_id": row["id"]})
    save_jsonl(dest / "candidates_with_teachers.jsonl", handoff)
    save_jsonl(dest / "priority_review.jsonl", priority)
    for owner, seed in [("you", 101), ("nayang", 202)]:
        packet = list(blind)
        random.Random(seed).shuffle(packet)
        save_jsonl(dest / f"reviewer_{owner}.jsonl", packet)
    save_jsonl(dest / "packet_id_mapping.jsonl", mapping)
    raw_quality = []
    for path in sorted((ROOT / "data").glob("first_diverse_*.jsonl")):
        values = read_jsonl(path)
        errors = []
        for row in values:
            try:
                validate([row])
            except ValueError as error:
                errors.append({"id": row.get("id"), "reason": str(error)})
        style = sum("不是" in r.get("user_input", "") for r in values)
        context_leak = [r["id"] for r in values if not r.get("history") and r.get("context", {}).get("active_task") == r.get("label")]
        raw_quality.append({"file": path.name, "n": len(values), "schema_failures": errors,
                            "repetitive_negation_count": style, "unjustified_task_context_ids": context_leak,
                            "decision": "quarantine_all_not_accepted", "jev_credits_spent_on_this_file": 0})
    save_json(dest / "raw_generation_quality.json", raw_quality)
    manifest = {"policy_version": POLICY["version"], "unique_candidates": len(rows),
                "by_proposed_label": dict(Counter(r['label'] for r in rows)),
                "jev_annotated_under_current_or_historical_policy": sum("jev" in r['teacher_annotations'] for r in handoff),
                "human_approved": 0, "review_pending": len(handoff),
                "raw_generated_quarantined": sum(r['n'] for r in raw_quality),
                "note": "Historical teacher labels require policy review; proposed labels are hidden in reviewer packets.",
                "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob("*.jsonl")}}
    save_json(dest / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
