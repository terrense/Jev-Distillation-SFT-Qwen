"""Merge two independently completed human review packets; disagreements stay pending."""
import argparse
from collections import Counter
from pathlib import Path
from intent_lab import LABELS, read_jsonl, save_json, save_jsonl, state_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-dir", default="data/handoff_v0.2")
    p.add_argument("--reviewer-a", required=True)
    p.add_argument("--reviewer-b", required=True)
    p.add_argument("--reviewer-a-name", default="you")
    p.add_argument("--reviewer-b-name", default="nayang")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.reviewer_a_name == a.reviewer_b_name:
        raise ValueError("two distinct reviewers are required")
    base = Path(a.packet_dir)
    mapping = {r["packet_id"]: r["original_id"] for r in read_jsonl(base / "packet_id_mapping.jsonl")}
    candidates = {r["id"]: r for r in read_jsonl(base / "candidates_with_teachers.jsonl")}
    def load(path):
        rows = read_jsonl(path)
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError("duplicate packet id")
        if set(r['id'] for r in rows) != set(mapping):
            raise ValueError("packet ids changed; labels must be joined by stable ids")
        return {r['id']: r for r in rows}
    left, right = load(a.reviewer_a), load(a.reviewer_b)
    accepted, unresolved, paired = [], [], []
    for packet_id, original_id in mapping.items():
        row, x, y = candidates[original_id], left[packet_id], right[packet_id]
        if state_hash(x) != state_hash(row) or state_hash(y) != state_hash(row):
            raise ValueError("reviewer changed input/context; regenerate and re-label that sample")
        lx, ly = x.get("label"), y.get("label")
        if lx is not None and lx not in LABELS or ly is not None and ly not in LABELS:
            raise ValueError("invalid review label")
        item = {**row, "reviewer_a_label": lx, "reviewer_b_label": ly,
                "reviewers": [a.reviewer_a_name, a.reviewer_b_name],
                "review_comments": [x.get("comment", ""), y.get("comment", "")]}
        if lx in LABELS and ly in LABELS:
            paired.append((lx, ly))
        if lx in LABELS and lx == ly and not x.get("action_if_unclear") and not y.get("action_if_unclear"):
            item.update(label=lx, adjudicated_label=lx, review_status="human_approved")
            accepted.append(item)
        else:
            item["review_status"] = "pending"
            unresolved.append(item)
    kappa = None
    observed = None
    if paired:
        n = len(paired)
        observed = sum(x == y for x, y in paired) / n
        cx, cy = Counter(x for x, _ in paired), Counter(y for _, y in paired)
        expected = sum(cx[label] * cy[label] for label in LABELS) / n**2
        kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    dest = Path(a.output)
    save_jsonl(dest / "human_approved.jsonl", accepted)
    save_jsonl(dest / "unresolved.jsonl", unresolved)
    stats = {"total": len(mapping), "jointly_labeled": len(paired), "human_approved": len(accepted),
             "unresolved": len(unresolved), "observed_agreement": observed, "cohen_kappa": kappa}
    save_json(dest / "review_stats.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
