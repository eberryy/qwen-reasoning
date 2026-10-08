"""Aggregate-only scorer for private puzzle truth and model predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metric import NOT_FOUND, extract_final_answer, has_boxed_answer, verify


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{path.name}:{line_number}: expected an object with string id")
            row_id = row["id"]
            if not row_id or row_id in rows:
                raise ValueError(f"{path.name}:{line_number}: empty or duplicate id")
            rows[row_id] = row
    if not rows:
        raise ValueError(f"{path.name}: no rows")
    return rows


def score(truth: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if truth.keys() != predictions.keys():
        raise ValueError(
            f"ID mismatch: {len(truth.keys() - predictions.keys())} missing predictions, "
            f"{len(predictions.keys() - truth.keys())} unexpected predictions"
        )
    totals = {"n": 0, "correct": 0, "boxed": 0, "invalid": 0}
    tasks: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "boxed": 0, "invalid": 0}
    )
    for row_id in sorted(truth):
        expected = truth[row_id]
        predicted = predictions[row_id]
        task = expected.get("task_type")
        answer = expected.get("answer")
        output = predicted.get("output")
        if not isinstance(task, str) or not task or answer is None or not isinstance(output, str):
            raise ValueError(f"Invalid truth/prediction fields for id {row_id!r}")
        extracted = extract_final_answer(output)
        flags = {
            "n": 1,
            "correct": int(verify(str(answer), extracted)),
            "boxed": int(has_boxed_answer(output)),
            "invalid": int(extracted == NOT_FOUND),
        }
        for key, value in flags.items():
            totals[key] += value
            tasks[task][key] += value
    return {
        "metric": "nvidia_nemotron_metric_v15_compatibility_port",
        "overall": {**totals, "accuracy": totals["correct"] / totals["n"]},
        "per_task": {
            name: {**values, "accuracy": values["correct"] / values["n"]}
            for name, values in sorted(tasks.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(read_jsonl(args.truth), read_jsonl(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Scored {report['overall']['n']} predictions; accuracy={report['overall']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
