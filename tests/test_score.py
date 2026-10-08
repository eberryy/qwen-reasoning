from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_reasoning.score import score  # noqa: E402


class ScoreTests(unittest.TestCase):
    def test_aggregate_only(self) -> None:
        truth = {
            "a": {"id": "a", "task_type": "equations", "answer": "42"},
            "b": {"id": "b", "task_type": "equations", "answer": "41"},
        }
        predictions = {
            "a": {"id": "a", "output": r"work \boxed{42}"},
            "b": {"id": "b", "output": "Final answer: 40"},
        }
        result = score(truth, predictions)
        self.assertEqual(result["overall"]["correct"], 1)
        self.assertEqual(result["overall"]["n"], 2)
        self.assertEqual(result["overall"]["boxed"], 1)
        self.assertNotIn("\"id\"", str(result))

    def test_missing_prediction_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            score({"a": {"id": "a"}}, {})


if __name__ == "__main__":
    unittest.main()
