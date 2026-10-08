# Qwen3-4B puzzle reasoning post-training

An evidence-backed SFT case study on six types of logic and symbolic puzzles. Starting from the same pinned Qwen3-4B-Base checkpoint, a 4-bit LoRA SFT run improved accuracy on a frozen 600-question validation set from **112/600 (18.67%) to 466/600 (77.67%)**, a **+59.0 percentage-point** difference.

| Task | Base | SFT |
| --- | ---: | ---: |
| Bit manipulation | 2/100 | 74/100 |
| Encryption | 0/100 | 58/100 |
| Equations | 7/100 | 45/100 |
| Gravity | 2/100 | 90/100 |
| Numeral systems | 66/100 | 99/100 |
| Unit conversion | 35/100 | 100/100 |

The SFT corpus contained 8,791 checked examples. It combined curated reasoning traces with rule-generated, answer-verified traces, then rejected conflicting prompt/answer pairs and excluded frozen evaluation questions. The encryption subset was replaced under an equal-example, equal-token control; gravity was repaired after an audit found the earlier data did not reliably teach the target behavior. The training objective was completion-only SFT on the same six-task distribution—not a claim that one rule or one data source alone produced the gain. [Data construction details](docs/DATA.md).

The experiment used TRL `SFTTrainer`, PEFT LoRA (`r=16`, `alpha=32`, all-linear), NF4 4-bit loading, bf16, 8,192-token training sequences, two epochs and four RTX 4090s. [The result card](results/final-validation.json) and [protocol](docs/PROTOCOL.md) give the exact comparison and its limits. The included [trainer](src/qwen_reasoning/train_sft.py) and [scorer](src/qwen_reasoning/score.py) are data-free reference implementations extracted from the experiment; the original run's private materializations and adapters are not distributed.

This is a **within-benchmark held-out comparison**, not evidence of broad out-of-domain transfer. Both models reached the 7,680-token generation cap on every validation item. The score gain does not establish improved natural stopping or inference efficiency. The development set was used to choose the SFT candidate and is not presented as a second independent test.

## Reproduce with your own licensed data

The training input is JSONL with `id`, `task_type`, `prompt` (system/user chat messages) and `completion` (assistant message). See [reproduction notes](docs/REPRODUCE.md). No private dataset, prompts, predictions, model weights, or server configuration is shipped here. The `results/` files contain aggregate counts only.

Code is Apache-2.0 licensed. The answer scorer is an attributed compatibility port of an Apache-2.0 public metric; the rule-generated training material was derived from a separately MIT-licensed public generator without vendoring its code. See [third-party notices](THIRD_PARTY_NOTICES.md).
