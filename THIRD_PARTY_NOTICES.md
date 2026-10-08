# Third-party notices

## Answer metric

`src/qwen_reasoning/metric.py` is adapted from the source repository's compatibility port of the public NVIDIA Nemotron Metric, Version 15 of 15, under Apache License 2.0. It retains answer extraction and comparison behavior, but omits model-loading and competition infrastructure. Source: https://www.kaggle.com/code/metric/nvidia-nemotron-metric . This provenance is included for license compliance; it does not imply that this repository trains or distributes NVIDIA model weights.

## Public puzzle generators

Part of the historical SFT data was materialized using selected generators from `livctr/nvidia-nemotron` at commit `597dcaaa9e7aebe81b61092260f9f1343c5a2fcc`, licensed MIT. Their code and generated rows are **not** included in this repository. Source: https://github.com/livctr/nvidia-nemotron . The dataset manifest in the original experiment records the precise source-file and materialization fingerprints.

## Base model

The experiment uses `Qwen/Qwen3-4B-Base` at revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`. No Qwen model weights are distributed here. See the upstream model repository for its own terms: https://huggingface.co/Qwen/Qwen3-4B-Base .
