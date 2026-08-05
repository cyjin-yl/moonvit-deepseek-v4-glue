from moonvit_glue.chat_contract import build_chat_prompt, build_chat_supervision


class FakeChatTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [1000 + ord(character) for character in text]


def test_visual_and_blind_chat_prompts_preserve_semantics_and_insert_one_id():
    tokenizer = FakeChatTokenizer()
    visual = build_chat_prompt(
        tokenizer,
        system_prompt="system rule",
        user_prompt="Target: Save",
        placeholder_token_id=151655,
        include_image=True,
    )
    blind = build_chat_prompt(
        tokenizer,
        system_prompt="system rule",
        user_prompt="Target: Save",
        placeholder_token_id=151655,
        include_image=False,
    )

    assert visual.input_ids.count(151655) == 1
    assert 151655 not in blind.input_ids
    assert visual.semantic_system_prompt == blind.semantic_system_prompt == "system rule"
    assert visual.semantic_user_prompt == blind.semantic_user_prompt == "Target: Save"
    assert visual.placeholder_count == 1
    assert blind.placeholder_count == 0
    assert visual.template_text_for_audit.count("<IMAGE_PLACEHOLDER>") == 1


def test_chat_supervision_masks_prompt_and_trains_exact_answer_plus_eos():
    tokenizer = FakeChatTokenizer()
    batch = build_chat_supervision(
        tokenizer,
        system_prompt="return a click",
        user_prompt="Target: Save",
        answer="click(start_box=[12, 34])",
        placeholder_token_id=151655,
        include_image=True,
        ignore_index=-100,
    )
    answer_ids = tokenizer.encode(
        "click(start_box=[12, 34])", add_special_tokens=False
    ) + [tokenizer.eos_token_id]

    assert batch.input_ids[: batch.prompt_length] == batch.prompt.input_ids
    assert batch.input_ids[batch.prompt_length :] == answer_ids
    assert batch.labels[: batch.prompt_length] == [-100] * batch.prompt_length
    assert batch.labels[batch.prompt_length :] == answer_ids
    assert batch.answer_tokens == len(answer_ids)
