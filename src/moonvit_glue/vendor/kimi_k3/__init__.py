"""MoonViT-V2 (Kimi K3 vision tower), vendored for standalone use.

Source: https://huggingface.co/moonshotai/Kimi-K3
License: Apache-2.0 (llava-derived parts) + Kimi K3 License (see LICENSE).

Only the vision tower is kept: ``MoonViT3dPretrainedModel`` plus its config
and image-processor helpers. The text model (``modeling_kimi_linear``) and
``KimiK3ForConditionalGeneration`` were removed, so this package imports
cleanly without the text-model dependency chain.
"""
