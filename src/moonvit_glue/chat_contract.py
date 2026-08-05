"""可迁移的 official-chat-template prompt 与 assistant-only supervision。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_IMAGE_SENTINEL = "\x00MOONVIT_IMAGE_PLACEHOLDER\x00"
_IMAGE_AUDIT_MARKER = "<IMAGE_PLACEHOLDER>"


@dataclass(frozen=True)
class ChatPrompt:
    input_ids: list[int]
    template_text_for_audit: str
    semantic_system_prompt: str
    semantic_user_prompt: str
    placeholder_count: int


@dataclass(frozen=True)
class ChatSupervision:
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    answer_tokens: int
    prompt: ChatPrompt


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded]


def build_chat_prompt(
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    placeholder_token_id: int,
    include_image: bool,
) -> ChatPrompt:
    """渲染 tokenizer 官方 chat template，并精确插入一个图像 token ID。

    placeholder 在模板渲染后按 token ID 插入，避免依赖 tokenizer 的文本切分，
    同一份语义消息也可由后续 DeepSeek receiver 复用。
    """

    placeholder = int(placeholder_token_id)
    if placeholder < 0:
        raise ValueError("placeholder_token_id must be non-negative")
    system = str(system_prompt)
    user = str(user_prompt)
    if _IMAGE_SENTINEL in system or _IMAGE_SENTINEL in user:
        raise ValueError("semantic prompt contains the reserved image sentinel")
    user_content = f"{_IMAGE_SENTINEL}\n{user}" if include_image else user
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("chat template must return text when tokenize=False")

    if include_image:
        if rendered.count(_IMAGE_SENTINEL) != 1:
            raise ValueError("chat template did not preserve exactly one image sentinel")
        before, after = rendered.split(_IMAGE_SENTINEL)
        input_ids = _encode(tokenizer, before) + [placeholder] + _encode(tokenizer, after)
        audit_text = rendered.replace(_IMAGE_SENTINEL, _IMAGE_AUDIT_MARKER)
        placeholder_count = 1
    else:
        if _IMAGE_SENTINEL in rendered:
            raise ValueError("blind chat template unexpectedly contains the image sentinel")
        input_ids = _encode(tokenizer, rendered)
        audit_text = rendered
        placeholder_count = 0
    actual_count = input_ids.count(placeholder)
    if actual_count != placeholder_count:
        raise ValueError(
            "semantic prompt collides with the image placeholder token ID: "
            f"expected {placeholder_count}, found {actual_count}"
        )
    return ChatPrompt(
        input_ids=input_ids,
        template_text_for_audit=audit_text,
        semantic_system_prompt=system,
        semantic_user_prompt=user,
        placeholder_count=placeholder_count,
    )


def build_chat_supervision(
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    answer: str,
    placeholder_token_id: int,
    include_image: bool,
    ignore_index: int = -100,
) -> ChatSupervision:
    """屏蔽完整官方 chat prompt，只监督精确答案与 im_end。"""

    prompt = build_chat_prompt(
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        placeholder_token_id=placeholder_token_id,
        include_image=include_image,
    )
    answer_ids = _encode(tokenizer, str(answer))
    if not answer_ids:
        raise ValueError("assistant answer must contain at least one token")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    supervised = answer_ids + [int(eos_token_id)]
    input_ids = prompt.input_ids + supervised
    labels = [int(ignore_index)] * len(prompt.input_ids) + supervised
    return ChatSupervision(
        input_ids=input_ids,
        labels=labels,
        prompt_length=len(prompt.input_ids),
        answer_tokens=len(supervised),
        prompt=prompt,
    )
