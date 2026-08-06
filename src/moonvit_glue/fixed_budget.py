"""固定预算代理训练的顺序、prompt 路由和恢复约束。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoutedTrainingExample:
    record_id: str
    prompt_route: str
    system_prompt: str
    user_prompt: str
    target_answer: str


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def feature_cache_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Resolve the expected frozen-tower/cache interface from a contract.

    The original Qwen3B contract only described the K3 V2 tower in prose, so
    the historical validator used ``1024`` and ``4`` literals.  Architecture
    controls can now provide a ``feature_cache_binding`` block (or equivalent
    fields under ``vision_tower``) without changing the training-order schema.
    Missing optional fields retain the legacy V2 defaults.
    """

    tower = contract.get("vision_tower", {})
    binding = contract.get("feature_cache_binding", {})
    if not isinstance(tower, dict) or not isinstance(binding, dict):
        raise ValueError("vision_tower and feature_cache_binding must be objects")

    def first(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    width = first(
        binding.get("vision_width"),
        tower.get("vision_width"),
        # Legacy K3 V2 is the only contract that predates explicit width.
        1024,
    )
    merge = first(binding.get("merge_factor"), tower.get("merge_factor"), 4)
    weight_sha = first(
        binding.get("moonvit_weights_sha256"),
        tower.get("extracted_weights_sha256"),
        tower.get("weights_sha256"),
        tower.get("model_weights_sha256"),
    )
    revision = first(
        binding.get("moonvit_revision"),
        tower.get("resolved_revision"),
        tower.get("source_resolved_revision"),
    )
    name = first(binding.get("vision_tower"), tower.get("name"))
    model = first(binding.get("moonvit_model"), tower.get("source_repo"))
    return {
        "vision_tower": str(name) if name is not None else None,
        "moonvit_model": str(model) if model is not None else None,
        "moonvit_revision": str(revision) if revision is not None else None,
        "moonvit_weights_sha256": str(weight_sha) if weight_sha is not None else None,
        "vision_width": int(width),
        "merge_factor": int(merge),
        "require_tower_identity": bool(binding.get("require_tower_identity", False)),
    }


def validate_fixed_budget_contract(
    contract: dict[str, Any],
    order_manifest: dict[str, Any],
    cache_manifest: dict[str, Any],
) -> dict[str, int]:
    """锁定 3B 主干、首个 budget、训练顺序和 frozen-MoonViT cache。"""

    if contract.get("schema_version") != "qwen25-3b-community-eval-contract-v1":
        raise ValueError("unexpected Qwen3B evaluation contract schema")
    proxy = contract["proxy_model"]
    if proxy.get("architecture") != "Qwen2ForCausalLM" or proxy.get("frozen") is not True:
        raise ValueError("proxy backbone must be the frozen pure-text Qwen2 model")
    if contract["vision_tower"].get("frozen") is not True:
        raise ValueError("MoonViT-V2 must remain frozen")

    cache_contract = feature_cache_contract(contract)
    projector = contract["canonical_projector"]
    receiver = contract["qwen_proxy_receiver"]
    _require_equal(
        int(projector["output_width"]),
        int(receiver["input_width"]),
        "projector and receiver widths differ",
    )
    _require_equal(
        int(receiver["output_width"]),
        int(proxy["hidden_size"]),
        "receiver and Qwen hidden widths differ",
    )
    _require_equal(
        int(receiver["trainable_parameter_count"]),
        0,
        "proxy receiver must remain parameter-free",
    )

    budget = contract["training_budget"]
    selection = order_manifest["selection"]
    expected_examples = int(budget["examples_seen_checkpoints"][0])
    expected_steps = int(budget["optimizer_steps_checkpoints"][0])
    matched_budget = {
        "examples_seen": expected_examples,
        "optimizer_steps": expected_steps,
        "micro_batch_size": int(budget["micro_batch_size"]),
        "gradient_accumulation": int(budget["gradient_accumulation"]),
        "real_global_batch": int(budget["real_global_batch"]),
    }
    for key, expected in matched_budget.items():
        _require_equal(int(selection[key]), expected, f"training-order {key} differs")
    _require_equal(
        matched_budget["micro_batch_size"],
        1,
        "formal Qwen3B trainer requires micro batch size 1",
    )
    _require_equal(
        matched_budget["micro_batch_size"]
        * matched_budget["gradient_accumulation"],
        matched_budget["real_global_batch"],
        "training budget batch arithmetic differs",
    )
    _require_equal(
        matched_budget["optimizer_steps"] * matched_budget["real_global_batch"],
        matched_budget["examples_seen"],
        "training budget examples/steps arithmetic differs",
    )
    _require_equal(
        int(selection["effective_epochs_denominator"]),
        int(budget["effective_epochs_denominator"]),
        "effective-epoch denominator differs",
    )
    if budget.get("feature_cache_allowed") is not True:
        raise ValueError("frozen feature cache is not allowed by the contract")
    if budget.get("activation_checkpointing") is not True:
        raise ValueError("activation checkpointing must remain enabled")
    _require_equal(budget.get("language_dtype"), "float16", "language dtype differs")
    _require_equal(budget.get("projector_dtype"), "float32", "projector dtype differs")

    order_records = order_manifest["records"]
    cache_records = cache_manifest["records"]
    _require_equal(len(order_records), expected_examples, "training-order count differs")
    _require_equal(int(cache_manifest["count"]), expected_examples, "cache count differs")
    _require_equal(len(cache_records), expected_examples, "cache record count differs")
    _require_equal(
        cache_manifest.get("training_order_manifest_sha256"),
        order_manifest.get("manifest_sha256"),
        "cache training-order manifest binding differs",
    )
    _require_equal(
        cache_manifest.get("training_order_records_sha256"),
        order_manifest.get("records_sha256"),
        "cache training-order record binding differs",
    )
    expected_cache_hash = cache_contract["moonvit_weights_sha256"]
    expected_cache_revision = cache_contract["moonvit_revision"]
    if expected_cache_hash is None and expected_cache_revision is None:
        raise ValueError("contract must pin a MoonViT revision or weight hash")
    if expected_cache_hash is not None:
        _require_equal(
            cache_manifest.get("moonvit_weights_sha256"),
            expected_cache_hash,
            "cache MoonViT weight identity differs",
        )
        _require_equal(
            order_manifest["feature_cache"].get("moonvit_weights_sha256"),
            expected_cache_hash,
            "training-order MoonViT weight identity differs",
        )
    if expected_cache_revision is not None and cache_contract["require_tower_identity"]:
        _require_equal(
            cache_manifest.get("moonvit_revision"),
            expected_cache_revision,
            "cache MoonViT resolved revision differs",
        )
        _require_equal(
            order_manifest["feature_cache"].get("moonvit_revision"),
            expected_cache_revision,
            "training-order MoonViT resolved revision differs",
        )
    if cache_contract["require_tower_identity"]:
        for source, label in ((cache_manifest, "cache"), (order_manifest["feature_cache"], "training-order")):
            _require_equal(
                source.get("vision_tower"),
                cache_contract["vision_tower"],
                f"{label} vision tower identity differs",
            )
    preprocessing = contract["image_preprocessing"]
    for source, label in (
        (order_manifest["feature_cache"], "training order"),
        (cache_manifest, "feature cache"),
    ):
        _require_equal(
            int(source["max_image_side"]),
            int(preprocessing["train_max_image_side"]),
            f"{label} max image side differs",
        )
        _require_equal(
            int(source["max_visual_tokens"]),
            int(preprocessing["train_max_visual_tokens"]),
            f"{label} visual-token budget differs",
        )
    _require_equal(
        int(cache_manifest["vision_width"]),
        cache_contract["vision_width"],
        "cache vision width differs",
    )
    _require_equal(
        int(cache_manifest["merge_factor"]),
        cache_contract["merge_factor"],
        "cache merge factor differs",
    )

    maximum_tokens = 0
    aliases = 0
    for index, (order_row, cache_row) in enumerate(
        zip(order_records, cache_records, strict=True)
    ):
        _require_equal(
            str(cache_row["id"]),
            str(order_row["id"]),
            f"cache order differs at row {index}",
        )
        shape = [int(value) for value in cache_row["feature_shape"]]
        if (
            len(shape) != 3
            or shape[1:] != [cache_contract["merge_factor"], cache_contract["vision_width"]]
            or shape[0] <= 0
        ):
            raise ValueError(f"cache feature shape differs at row {index}")
        maximum_tokens = max(maximum_tokens, shape[0])
        if shape[0] > int(preprocessing["train_max_visual_tokens"]):
            raise ValueError(f"cache row exceeds the visual-token budget: {index}")
        _require_equal(cache_row["dtype"], "float32", f"cache dtype differs at row {index}")
        aliases += int("alias_of" in cache_row)

    _require_equal(
        aliases,
        int(cache_manifest["aliased_records"]),
        "cache alias count differs",
    )
    unique_spans = int(cache_manifest["unique_feature_spans"])
    _require_equal(
        unique_spans,
        expected_examples - aliases,
        "cache unique-span arithmetic differs",
    )
    _require_equal(
        unique_spans,
        int(order_manifest["unique_image_sha256"]),
        "cache unique spans differ from training-order image identity",
    )
    return {
        **matched_budget,
        "unique_feature_spans": unique_spans,
        "aliased_records": aliases,
        "maximum_visual_tokens": maximum_tokens,
        "projector_output_width": int(projector["output_width"]),
        "receiver_output_width": int(receiver["output_width"]),
    }


def fixed_batch_record_indices(
    *, optimizer_step: int, total_examples: int, gradient_accumulation: int
) -> list[int]:
    """返回零基 optimizer step 对应的连续记录，不循环、不 shuffle。"""

    step = int(optimizer_step)
    total = int(total_examples)
    accumulation = int(gradient_accumulation)
    if step < 0 or total <= 0 or accumulation <= 0:
        raise ValueError("optimizer step, total examples and accumulation are invalid")
    start = step * accumulation
    end = start + accumulation
    if end > total:
        raise ValueError("optimizer step falls outside the frozen order")
    return list(range(start, end))


def validate_resume_history(
    *,
    start_step: int,
    history: list[dict[str, Any]],
    total_examples: int,
    gradient_accumulation: int,
) -> dict[str, int]:
    """确保 checkpoint 恢复点恰好对应固定顺序的下一个样本。"""

    step = int(start_step)
    total = int(total_examples)
    accumulation = int(gradient_accumulation)
    if step < 0 or step * accumulation > total:
        raise ValueError("resume step falls outside the frozen order")
    if len(history) != step:
        raise ValueError("resume history length differs from optimizer step")
    previous_answer_tokens = 0
    for expected_step, row in enumerate(history, start=1):
        if int(row.get("step", -1)) != expected_step:
            raise ValueError("resume history optimizer steps are not contiguous")
        expected_examples = expected_step * accumulation
        if int(row.get("examples_seen", -1)) != expected_examples:
            raise ValueError("resume history examples_seen differs from fixed order")
        answer_tokens = int(row.get("answer_tokens_seen", -1))
        if answer_tokens < previous_answer_tokens:
            raise ValueError("resume history answer_tokens_seen is not monotonic")
        previous_answer_tokens = answer_tokens
    return {
        "next_example_index": step * accumulation,
        "examples_seen": step * accumulation,
        "answer_tokens_seen": previous_answer_tokens,
    }


def route_training_example(
    contract: dict[str, Any],
    order_entry: dict[str, Any],
    record: dict[str, Any],
) -> RoutedTrainingExample:
    """按冻结 route 生成语义 prompt，并只采用 manifest 中的 teacher target。"""

    record_id = str(order_entry["id"])
    if str(record.get("id")) != record_id:
        raise ValueError("training record and order entry IDs differ")
    question = str(record.get("question", ""))
    target = str(order_entry.get("target_answer", ""))
    if not question or not target:
        raise ValueError("training question and target must be non-empty")
    prompts = contract["prompt_and_generation"]
    route = str(order_entry["prompt_route"])
    if route == "grounding":
        system_prompt = str(prompts["system_prompt"])
        user_prompt = str(prompts["user_prompt"]).format(instruction=question)
    elif route == "short_answer":
        system_prompt = str(prompts["short_answer_system_prompt"])
        user_prompt = str(prompts["short_answer_user_prompt"]).format(question=question)
    else:
        raise ValueError(f"unknown prompt route: {route}")
    return RoutedTrainingExample(
        record_id=record_id,
        prompt_route=route,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        target_answer=target,
    )
