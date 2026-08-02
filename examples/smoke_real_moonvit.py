"""Use the real standalone MoonViT with a tiny, text-only GPT-2 checkpoint."""

import argparse

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from moonvit_glue import (
    MoonViTEncoder,
    PatchMergerProjector,
    ProjectorConfig,
    VisionCausalLM,
)


parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--text-model", default="sshleifer/tiny-gpt2")
args = parser.parse_args()

moonvit = MoonViTEncoder.from_pretrained(torch_dtype="auto")
image_inputs = moonvit.preprocess(Image.open(args.image).convert("RGB"))
image_feature_groups = moonvit(**image_inputs)

tokenizer = AutoTokenizer.from_pretrained(args.text_model)
language_model = AutoModelForCausalLM.from_pretrained(args.text_model)
placeholder_id = tokenizer.eos_token_id
prefix = tokenizer.encode("Image:", add_special_tokens=False)
suffix = tokenizer.encode(" Describe it.", add_special_tokens=False)
input_ids = torch.tensor([[*prefix, placeholder_id, *suffix]])
labels = input_ids.clone()
labels[input_ids == placeholder_id] = -100

projector = PatchMergerProjector(
    ProjectorConfig(
        vision_width=moonvit.vision_width,
        language_width=language_model.config.hidden_size,
        merge_factor=moonvit.merge_factor,
    )
)
model = VisionCausalLM(
    language_model=language_model,
    projector=projector,
    placeholder_token_id=placeholder_id,
    backbone_kind="generic",
)
outputs = model(
    input_ids=input_ids,
    image_feature_groups=image_feature_groups,
    labels=labels,
)
outputs.loss.backward()
print(f"MoonViT groups: {[tuple(item.shape) for item in image_feature_groups]}")
print(f"Expanded sequence: {outputs.logits.shape[1]} tokens")
print(f"Loss: {outputs.loss.item():.4f}")

