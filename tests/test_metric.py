from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_reasoning.metric import (  # noqa: E402
    NOT_FOUND,
    extract_final_answer,
    grade,
    has_boxed_answer,
    verify,
)


class ExtractFinalAnswerTests(unittest.TestCase):
    def test_none_and_empty(self) -> None:
        self.assertEqual(extract_final_answer(None), NOT_FOUND)
        self.assertEqual(extract_final_answer(" \n "), NOT_FOUND)

    def test_last_non_empty_box_wins(self) -> None:
        text = r"first \boxed{41}; empty \boxed{}; final \boxed{42}"
        self.assertEqual(extract_final_answer(text), "42")

    def test_nested_latex_is_preserved(self) -> None:
        self.assertEqual(
            extract_final_answer(r"answer \boxed{\frac{1}{2}}"),
            r"\frac{1}{2}",
        )

    def test_literal_closing_brace_in_answer(self) -> None:
        self.assertEqual(extract_final_answer(r"answer \boxed{}52}"), "}52")

    def test_official_fallback_order(self) -> None:
        self.assertEqual(extract_final_answer("Final answer: XLVII"), "XLVII")
        self.assertEqual(extract_final_answer("numbers 12 then -3.5"), "-3.5")
        self.assertEqual(extract_final_answer("alpha\nomega"), "omega")

    def test_fullwidth_colon_final_answer(self) -> None:
        self.assertEqual(extract_final_answer("Final answer：XLVII"), "XLVII")


class VerifyTests(unittest.TestCase):
    def test_binary_strings_are_strict(self) -> None:
        self.assertTrue(verify("10011000", "10011000"))
        self.assertFalse(verify("10011000", "10011001"))
        self.assertFalse(verify("11011", "00011011"))
        # The official branch treats every all-0/1 string as binary, including
        # a value that might otherwise be interpreted as decimal one hundred.
        self.assertFalse(verify("100", "101"))

    def test_numeric_tolerance(self) -> None:
        self.assertTrue(verify("24.64", "24.6401"))
        self.assertTrue(verify("200", "202"))
        self.assertFalse(verify("200", "202.1"))

    def test_string_comparison_is_case_insensitive(self) -> None:
        self.assertTrue(verify("XLVII", "xlvii"))

    def test_grade_and_format(self) -> None:
        self.assertTrue(grade(r"work \boxed{42}", "42"))
        self.assertTrue(has_boxed_answer(r"work \boxed{42}"))
        self.assertFalse(has_boxed_answer("Final answer: 42"))
        self.assertFalse(has_boxed_answer(r"work \boxed{}"))


if __name__ == "__main__":
    unittest.main()
