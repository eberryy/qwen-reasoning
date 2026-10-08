# Data construction and controls

The final SFT materialization had 8,791 rows across six puzzle families. Early examples used curated English chain-of-thought traces. For tasks where that coverage was weak or inconsistent, rule-based generators supplied more structurally aligned traces: a solver generated the answer, its explanation was checked against the answer, and rows failing verification were rejected. This reduced dependence on unverified teacher-model prose for exact symbolic and arithmetic manipulations. The upstream generator provenance is disclosed in `THIRD_PARTY_NOTICES.md`; neither its source nor the generated private rows are redistributed here.

| Task | Rows | Construction note |
| --- | ---: | --- |
| Bit manipulation | 1,602 | Rule-generated checked traces for bit-level operations |
| Encryption | 1,323 | 439 public-generator rows replaced 439 historical rows, keeping both count and total training tokens fixed |
| Equations | 1,555 | 732 numeric and 823 symbolic rule-generated checked traces |
| Gravity | 1,441 | Historical gravity rows were replaced after an error audit |
| Numeral systems | 1,426 | Curated historical traces retained |
| Unit conversion | 1,444 | Curated historical traces retained |

The construction pipeline rejected inconsistent same-prompt answers, deduplicated semantically equivalent prompts, excluded development/validation prompts, and required prompt-only solvability (the answer could not be inferred only from hidden generator state). The final tokenizer audit found no sequence over 8,192 tokens; longest was 7,635. Dataset fingerprint: `014e62b5486f3e32942194f19142114bba165b9290149c9165e2ab83334a61b2`.

The equal-count/equal-token encryption replacement controls dataset size for that particular design decision. The **Base→SFT** final comparison nevertheless evaluates the whole treatment; it does not isolate the encryption replacement, solver traces or any individual component. The original materialization is unavailable publicly, so this repository provides a data contract and evidence card rather than asserting fully push-button dataset reproduction.
