import unittest

from evals.suites.memory.evidence import bound_evidence, objects, serialized_size


class EvidenceBudgetTests(unittest.TestCase):
    def test_truncated_text_retains_source_identity_and_explicit_flag(self) -> None:
        source = {"id": 1, "chat_key": "a", "date": "2026-01-01", "text": "x" * 8000}
        original = {"sources": [source]}
        result = bound_evidence(original, 1000)
        self.assertLessEqual(serialized_size(result), 1000)
        self.assertTrue(result["evidence_truncated"])
        self.assertEqual(result["sources"][0]["id"], 1)
        self.assertTrue(result["sources"][0]["truncated"])
        self.assertEqual(source["text"], "x" * 8000)

    def test_removed_sources_cannot_appear_in_exposed_object_walk(self) -> None:
        original = {
            "messages": [
                {"id": i, "chat_key": "a", "date": "2026-01-01", "text": "x" * 150}
                for i in range(40)
            ]
        }
        result = bound_evidence(original, 1200)
        self.assertLessEqual(serialized_size(result), 1200)
        exposed = {v["id"] for v in objects(result) if "id" in v}
        self.assertLess(len(exposed), 40)
        self.assertEqual(exposed, {v["id"] for v in result["messages"]})

    def test_exhausted_budget_returns_no_source_records(self) -> None:
        result = bound_evidence({"source": {"id": 1, "text": "x" * 500}}, 0)
        self.assertIn("error", result)
        self.assertFalse(any("id" in item for item in objects(result)))
