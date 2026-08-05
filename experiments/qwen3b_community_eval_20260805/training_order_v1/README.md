# Qwen2.5-3B 4k training-order freeze

This package freezes the first matched-budget training prefix for the pure-text
`Qwen/Qwen2.5-3B-Instruct` proxy before any 4k optimization result exists. The
selection is the first 4,000 rows of the Package 15A training pack in source
order, with no shuffle and no held-out-row removal. At micro batch 1 and
gradient accumulation 8, this is exactly 500 optimizer steps, one subset pass,
and 0.0675698503 effective epochs against the full 59,198-row pack.

`MANIFEST.json` binds every logical record, question, raw answer list, actual
teacher target, source route, image path, encoded-image SHA-256, byte count and
dimensions. It self-hashes to
`ddca738e366f37237354bb011bdff1a00d010bdf256ef9101a6adbf35ab9c2fd`;
the ordered record list hashes to
`61fa7360208b90bb791914c27801cc90d702d579155d798d39ae4e400f7f315e`.
The prefix contains 1,985 TextVQA, 1,160 DocVQA, 516 OCRBench-derived and 339
ShowUI desktop rows. All 4,000 IDs and image paths are unique; 3,534 unique
image byte hashes show that some distinct training rows intentionally share an
identical source image.

The training pack predates the strict click-output contract. All 339 ShowUI
answers use the legacy no-space spelling and are deterministically rewritten to
the sole accepted `click(start_box=[x, y])` target. Raw answer-list hashes remain
unchanged. Two TextVQA rows have `(` and `a` as majority answers; VQA
normalization erases both. Those targets use a deterministic raw-string majority
fallback. The manifest records the transform for every row.

`INDEPENDENT_VERIFICATION.json` re-read the full training JSONL and rehashed all
4,000 logical records, teacher targets and 1,523,324,154 referenced image bytes.
It also reopened every image and checked dimensions. All 4,000 records, images
and targets matched with no mismatch.

Two failed freeze attempts remain immutable under `failures/`. The first exposed
the legacy click spacing. The second exposed punctuation-only TextVQA targets.
Each failure stopped before a manifest or training result was written.

The complete canonical root remains on the V100 workstation at
`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/training_order_v1_retry2`.
This package establishes data/order identity only. It contains no trained 3B
checkpoint, capability score or final-half evaluation. Its DeepSeek migration
label is `directly_transferable`: the same ordered records, canonical targets,
image identities and examples-seen accounting can drive either frozen text
backbone.
