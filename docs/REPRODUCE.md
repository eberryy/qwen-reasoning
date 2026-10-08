# Reproduction boundary

This is a compact public reference, not an upload of the original private materializations. To rerun SFT, supply your own licensed JSONL records with one object per line:

```json
{"id":"example-1","task_type":"unit_conversion","prompt":[{"role":"user","content":"Convert 2 km to m."}],"completion":[{"role":"assistant","content":"2 km is 2000 m. Therefore \\boxed{2000}."}]}
```

Install the exact training package versions on a CUDA-compatible machine, including the PyTorch 2.6.0+cu124 build from the official PyTorch index. The original run additionally used flash-attn 2.7.4.post1, a four-GPU DDP launch and a pinned Qwen model revision. The lightweight training entry point preserves the key TRL/PEFT settings but does not reimplement all original artifact gates and distributed recovery logic. Dry-run it on a small dataset before a full run.

```bash
python -m pip install -e '.[train]'
accelerate launch --num_processes 4 -m qwen_reasoning.train_sft --dataset /path/to/train.jsonl --output /path/to/output
```

To score already-generated predictions, supply truth rows with `id`, `task_type`, `answer` and prediction rows with `id`, `output`:

```bash
qwen-reasoning-score --truth /path/to/truth.jsonl --predictions /path/to/predictions.jsonl --output /path/to/aggregate.json
```

The scorer writes aggregate counts only; it refuses duplicate IDs and incomplete or extra predictions. Do not treat its output as the historical reported result unless your inputs, model checkpoint and generation protocol match the frozen experiment.
