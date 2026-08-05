# Qwen2.5-3B grounding-enriched 4k training-order freeze

Package 15I freezes the single-variable data screen selected after Packages
15G/15H rejected the first 3B checkpoint. It was created before any candidate
training output. The model, exact step0 projector, 4,000-example budget, 500
optimizer steps, image preprocessing, fixed receiver and evaluators remain
unchanged. The treatment raises explicit ShowUI click supervision from
339/4,000 rows in the baseline prefix to 2,000/4,000 rows.

Selection is deterministic. The freezer takes the first 2,000 grounding rows
and first 2,000 short-answer rows from the immutable 59,198-row source pack,
preserves source order within each route, then alternates grounding first and
short answer second. Every real global batch of eight therefore contains four
rows from each route. Source counts are 2,000 ShowUI desktop, 1,080 TextVQA,
649 DocVQA and 271 OCRBench-derived `train` rows.

`MANIFEST.json` binds every source row index, logical record, question, raw
answer list, canonical teacher target, image path, encoded-image SHA-256, byte
count and dimensions. Its self-hash is
`d632ecc2c9bc216a552f240e87b9733904b67dcfe30c489a62ba03df25370bf1`;
the ordered record list hashes to
`f3c3dec199a30927fb715b2d4fbc890baa8e3f3a456b56707f46490164b915ab`.
The 4,000 paths resolve to 2,013 unique image byte hashes because multiple
grounding instructions intentionally share screenshots.

`INDEPENDENT_VERIFICATION.json` independently re-read the full JSONL and
matched 4,000/4,000 logical records, teacher targets, images and dimensions.
It rehashed 1,255,969,179 referenced image bytes and reconstructed the exact
first-N-per-route selection from source data. Every registered check passed.

The canonical root remains on the local V100 workstation at
`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/training_order_grounding_enriched_4k_v1`.
Runner commit `c43c161a084b9446d35da12ad667d2fe42e4f3a7` passed the complete V100
repository suite. This package contains no trained candidate, capability score
or final-half evaluation. No paid resource was used. Its DeepSeek migration
label is `directly_transferable` because selection, order, targets, image
identity and examples-seen accounting are backbone-independent.
