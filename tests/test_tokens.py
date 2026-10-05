"""Token counting with the one tokenizer every budget uses."""

from cqa.tokens import count_tokens


def test_known_counts():
    assert count_tokens("") == 0
    assert count_tokens("hello world") == 2


def test_special_token_text_is_ordinary_text():
    assert count_tokens("<|endoftext|>") > 1
