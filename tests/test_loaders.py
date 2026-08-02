import pytest

from moonvit_glue.loaders import resolve_placeholder_token_id


class FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def test_existing_deepseek_image_token_is_selected_without_resizing_vocab():
    tokenizer = FakeTokenizer({"hello": 1, "<｜image｜>": 129279})

    assert resolve_placeholder_token_id(tokenizer, "<｜image｜>") == 129279


def test_missing_placeholder_never_silently_adds_a_new_token():
    with pytest.raises(ValueError, match="must already exist"):
        resolve_placeholder_token_id(FakeTokenizer({"hello": 1}), "<image>")
