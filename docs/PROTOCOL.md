# Frozen comparison protocol

The candidate SFT checkpoint was selected using development data before the 600-question validation labels were inspected. Validation has 100 items from each of six puzzle tasks. Base and SFT use the identical Qwen3-4B-Base revision, question set, prompt template, 4-bit inference mode, greedy decoding and answer metric. Training and validation IDs are disjoint; the source dataset manifest recorded zero semantic prompt overlap. A single validation confirmation was run.

Training used 8,791 examples. The data pipeline checked answers with deterministic solvers, rejected conflicting same-prompt answers, applied semantic deduplication, excluded frozen evaluation prompts, and verified each chat-templated sequence was at most 8,192 tokens. The historical training-data artifact is not redistributed. Its SHA256 fingerprint and task counts appear in `results/training.json` to identify the run, not to imply that a reader can reproduce the dataset from this repository alone.

The scorer follows the public metric's last-nonempty-box extraction precedence, fallback extraction and numeric/string comparison. The metric provenance is in `THIRD_PARTY_NOTICES.md`. The public `score.py` can independently evaluate any licensed truth/prediction JSONL with the same field contract.

The reported +59.0 percentage points is the **combined** data-and-SFT treatment versus the untrained base. It is not an ablation identifying the contribution of one source, task or training trick. All 1,200 generations (600 per model) hit the 7,680-token cap; final-answer format rate increased but natural stopping did not improve in this test. The development split was used for selection and must not be counted as independent validation.
