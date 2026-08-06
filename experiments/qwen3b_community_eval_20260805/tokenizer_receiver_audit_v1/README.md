# Tokenizer and receiver audit

This artifact records the pinned tokenizer/config identities and the special
embedding rows that matter to the V1/V2 receiver screen. It deliberately does
not copy any model shard into Git. Qwen2.5 is the primary pure-text proxy;
Qwen3.5 is a native-VLM receiver-prior diagnostic; DeepSeek is the final target.

The Qwen2.5 image-token rows are replaced by `VisionCausalLM` projector output
at runtime. Their presence in the vocabulary therefore proves interface
compatibility only. DeepSeek keeps its image IDs in `input_ids` for Hash-MoE
routing while the embedding lookup output is overridden.
