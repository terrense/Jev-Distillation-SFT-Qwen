"""Offline checks of failure accounting and label/leakage safeguards; no API calls."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import intent_lab as lab


class PipelineChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def row(self, ident="a", label="triage", status="human_approved", family="f1", text="头疼挂什么科"):
        return {"id": ident, "family_id": family, "user_input": text, "label": label,
                "context": {"stage": "unknown", "active_task": None}, "history": [],
                "tags": [], "source": "test_fixture", "review_status": status,
                "policy_version": lab.POLICY["version"]}

    def save(self, name, rows):
        path = self.root / name
        lab.save_jsonl(path, rows)
        return str(path)

    def call(self, function, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            function(argparse.Namespace(**kwargs))

    def test_pending_labels_cannot_be_exported(self):
        with self.assertRaises(ValueError):
            self.call(lab.export_sft, input=self.save("pending.jsonl", [self.row(status="pending")]), output=str(self.root / "sft.jsonl"))

    def test_dependency_failure_is_not_a_valid_other_classification(self):
        row = self.row()
        gold = self.save("gold.jsonl", [row])
        predictions = self.save("pred.jsonl", [{"id": "a", "state_hash": lab.state_hash(row), "error": "HTTP_503"}])
        output = str(self.root / "score.json")
        self.call(lab.score, gold=gold, predictions=predictions, output=output, allow_provisional=False)
        result = json.loads(Path(output).read_text(encoding="utf-8"))
        self.assertEqual(result["confusion"]["triage"]["ERROR"], 1)
        self.assertEqual(result["confusion"]["triage"]["other"], 0)
        self.assertEqual(result["accuracy"], 0)

    def test_predictions_for_a_different_context_are_rejected(self):
        row = self.row()
        bad = {"id": "a", "state_hash": "stale", "prediction": "triage", "format_valid": True}
        with self.assertRaises(ValueError):
            self.call(lab.score, gold=self.save("gold.jsonl", [row]), predictions=self.save("pred.jsonl", [bad]),
                      output=str(self.root / "score.json"), allow_provisional=False)

    def test_family_and_duplicate_text_stay_in_one_partition(self):
        rows = [self.row("a", family="same", text="头疼挂什么科"),
                self.row("b", family="same", text="我头晕看哪科"),
                self.row("c", family="different", text="我头晕看哪科")]
        dest = self.root / "splits"
        self.call(lab.split, input=self.save("rows.jsonl", rows), output=str(dest), seed=42, seed_only=False)
        occupied = [name for name in ["train", "dev", "audit_holdout"] if lab.read_jsonl(dest / (name + ".jsonl"))]
        self.assertEqual(len(occupied), 1)
        self.assertEqual(len(lab.read_jsonl(dest / (occupied[0] + ".jsonl"))), 3)

    def test_low_probability_consensus_is_queued_for_review(self):
        row = self.row(status="pending")
        base = {"id": "a", "state_hash": lab.state_hash(row), "prediction": "triage", "format_valid": True}
        a = {**base, "provider": "jev", "teacher_model": "jev-test", "answer": {"probabilities": {"triage": .6, "info": .4}}}
        b = {**base, "provider": "compat", "teacher_model": "different-test"}
        dest = self.root / "consensus"
        self.call(lab.consensus, input=self.save("rows.jsonl", [row]), teacher_a=self.save("a.jsonl", [a]),
                  teacher_b=self.save("b.jsonl", [b]), output=str(dest), jev_min_prob=.9, jev_min_margin=.3)
        self.assertEqual(lab.read_jsonl(dest / "accepted.jsonl"), [])
        self.assertEqual(len(lab.read_jsonl(dest / "human_review_queue.jsonl")), 1)

    def test_two_calls_to_same_model_are_not_independent(self):
        row = self.row(status="pending")
        pred = {"id": "a", "state_hash": lab.state_hash(row), "prediction": "triage", "format_valid": True, "teacher_model": "same-model"}
        with self.assertRaises(ValueError):
            self.call(lab.consensus, input=self.save("rows.jsonl", [row]), teacher_a=self.save("a.jsonl", [pred]),
                      teacher_b=self.save("b.jsonl", [pred]), output=str(self.root / "out"), jev_min_prob=.9, jev_min_margin=.3)

    def test_existing_conflicting_splits_cannot_silently_be_reshuffled(self):
        a, b = self.row("a"), self.row("b", text="另一个说法")
        a["split"], b["split"] = "train", "dev"
        with self.assertRaises(ValueError):
            self.call(lab.split, input=self.save("rows.jsonl", [a, b]), output=str(self.root / "out"), seed=42, seed_only=False)

    def test_history_without_speaker_roles_is_rejected_before_teacher_calls(self):
        row = self.row()
        row["history"] = ["没有role的问句不能冒充对话"]
        with self.assertRaises(ValueError):
            lab.validate([row])


if __name__ == "__main__":
    unittest.main()
