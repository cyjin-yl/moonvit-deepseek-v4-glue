"""MoonViT-to-causal-LM glue components.

Torch-dependent symbols are imported lazily so that the pure-Python
``moonvit_glue.metrics`` module stays importable on machines without a
PyTorch install (for example, benchmark-scoring-only environments).
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "CheckpointUploader": "checkpointing",
    "load_training_checkpoint": "checkpointing",
    "save_training_checkpoint": "checkpointing",
    "DEEPSEEK_FLASH_0731": "loaders",
    "DEFAULT_IMAGE_TOKEN": "loaders",
    "LoadedVisionLM": "loaders",
    "load_deepseek_flash_0731": "loaders",
    "resolve_placeholder_token_id": "loaders",
    "MultimodalInputs": "merge",
    "expand_image_placeholders": "merge",
    "VisionCausalLM": "model",
    "MoonViTEncoder": "moonvit",
    "build_moonvit_v2": "moonvit_v2",
    "load_moonvit_v2_encoder": "moonvit_v2",
    "load_moonvit_v2_processor": "moonvit_v2",
    "register_sdpa_attention": "moonvit_v2",
    "PatchMergerProjector": "projector",
    "ProjectorConfig": "projector",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
