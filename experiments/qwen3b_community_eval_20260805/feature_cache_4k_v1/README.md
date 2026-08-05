# Qwen2.5-3B 4k MoonViT feature cache

This package records the canonical frozen-MoonViT feature cache for the exact
Package 15C 4,000-row training order. It was generated on the local V100 from a
clean worktree at Git commit
`1e4c4000a88b02761abc5051ea78af9b2c7d4142`, with MoonViT-V2 SHA-256
`01436a95939965185bb853ddf984e09c00f597b9c2f6708ba302ffbaf75ced24`,
eager attention, float32 feature storage, maximum image side 448 and at most 256
visual groups per sample.

The run cached all 4,000 rows with zero failures in 503.5901 seconds and used
1,949,755,904 peak GPU bytes. Content addressing reduced the actual MoonViT
forwards to 3,534: 466 later rows reused the first occurrence of the same image
byte hash. The 111 safetensors shards contain 10,372,103,792 bytes and remain on
the V100 workstation. They are intentionally omitted from Git; the checked-in
remote artifact manifest binds every full-cache file and shard by size and
SHA-256.

`INDEPENDENT_VERIFICATION.json` was produced by the separately committed
verifier at `a9bd07b97e9bfc11ae4b82d69584a88b8799a646`. It rehashed all 111 shards,
read all 4,000 logical records, rejected non-finite or shape-mismatched tensors,
checked 3,534 unique spans and 466 aliases, rehashed the recorded runtime source
files, and matched every row to training-order manifest
`ddca738e366f37237354bb011bdff1a00d010bdf256ef9101a6adbf35ab9c2fd`.
The largest cached visual sequence is exactly 256 groups.

The first attempt is retained under `failures/`. It reached 1,128 rows before a
provenance audit found that the runner was executing uncommitted files while
reporting an older HEAD. The process was stopped, the partial root was preserved
without a final cache manifest, and a new clean-worktree run was started in a
fresh directory. The failed root must never be used for training.

The canonical full cache remains at
`/run/media/ezra/13D010B6FDBC1A06/data/qwen3b_contract/feature_cache_4k_v1_retry1`.
This is an engineering and data-identity result. It contains no trained Qwen3B
checkpoint, grounding score, paired-preference result or final-half evaluation.
Its DeepSeek migration label is `directly_transferable`: the frozen MoonViT
representations and exact record order do not depend on Qwen internals.
