"""Compatibility port of the public NVIDIA Nemotron Metric v15.

Source:
https://www.kaggle.com/code/metric/nvidia-nemotron-metric

The source notebook is published under Apache License 2.0. Only the answer
extraction and verification behavior is retained here; Kaggle model-loading
and submission infrastructure is intentionally omitted.
"""

from __future__ import annotations

import math
import re

METRIC_NAME = "nvidia_nemotron_metric_v15"
NOT_FOUND = "NOT_FOUND"


def extract_final_answer(text: str | None) -> str:
    r"""Extract the answer using the ordering in the official metric notebook."""
    if text is None:
        return NOT_FOUND

    # For each \boxed{ occurrence, take everything up to the final closing
    # brace before the next box (or end of text). This preserves nested LaTeX
    # and answers that themselves contain a literal closing brace.
    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches: list[str] = []
    for index, match in enumerate(boxed_starts):
        start = match.end()
        end = (
            boxed_starts[index + 1].start()
            if index + 1 < len(boxed_starts)
            else len(text)
        )
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)

    if matches:
        non_empty = [match.strip() for match in matches if match.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            return found[-1].strip()

    found = re.findall(r"-?\d+(?:\.\d+)?", text)
    if found:
        return found[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else NOT_FOUND


def verify(stored_answer: str, predicted: str) -> bool:
    """Compare an extracted prediction with a ground-truth answer."""
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(
            stored_num,
            predicted_num,
            rel_tol=1e-2,
            abs_tol=1e-5,
        )
    except Exception:
        return predicted.lower() == stored_answer.lower()


def grade(generated_text: str | None, ground_truth: str) -> bool:
    """Extract and verify a full model generation."""
    return verify(str(ground_truth), extract_final_answer(generated_text))


def has_boxed_answer(generated_text: str | None) -> bool:
    """Return whether a generation contains a non-empty official boxed answer."""
    if generated_text is None or r"\boxed{" not in generated_text:
        return False
    boxed = extract_final_answer(generated_text)
    return boxed not in {"", NOT_FOUND}
