"""Download configs/tokenizer only; do not download either model's weights."""

from transformers import AutoConfig, AutoTokenizer

from moonvit_glue import DEFAULT_IMAGE_TOKEN, ProjectorConfig, resolve_placeholder_token_id


moonvit = AutoConfig.from_pretrained(
    "moonshotai/MoonViT-SO-400M", trust_remote_code=True
)
deepseek = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V4-Flash-0731")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Flash-0731")
merge_kernel = tuple(moonvit.merge_kernel_size)
projector_config = ProjectorConfig(
    vision_width=moonvit.hidden_size,
    language_width=deepseek.hidden_size,
    merge_factor=merge_kernel[0] * merge_kernel[1],
)
parameter_count = (
    2 * projector_config.vision_width
    + projector_config.flattened_vision_width
    * projector_config.effective_projector_width
    + projector_config.effective_projector_width
    + projector_config.effective_projector_width * projector_config.language_width
    + projector_config.language_width
)

print(f"MoonViT width: {projector_config.vision_width}")
print(f"Patch merge: {merge_kernel} -> factor {projector_config.merge_factor}")
print(f"DeepSeek hidden size: {projector_config.language_width}")
print(f"Projector hidden size: {projector_config.effective_projector_width}")
print(f"Projector parameters: {parameter_count:,}")
print(
    f"Placeholder: {DEFAULT_IMAGE_TOKEN!r} -> "
    f"{resolve_placeholder_token_id(tokenizer)}"
)

