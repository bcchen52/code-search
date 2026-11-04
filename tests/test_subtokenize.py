"""The code subtokenizer used for lexical search."""

import pytest

from cqa.chunking.subtokenize import lexical_text, split_identifier, tokenize_code, tokenize_path


@pytest.mark.parametrize(
    "word, expected",
    [
        ("parseHTTPHeader", ["parse", "http", "header", "parsehttpheader"]),
        ("MAX_RETRY_COUNT", ["max", "retry", "count", "max_retry_count"]),
        ("HTTPServer", ["http", "server", "httpserver"]),
        ("getUTF8String", ["get", "utf8", "string", "getutf8string"]),
        ("parse_v2_header", ["parse", "v2", "header", "parse_v2_header"]),
        ("_hash_token", ["hash", "token", "_hash_token"]),
        ("session", ["session"]),
    ],
)
def test_split_identifier(word, expected):
    assert split_identifier(word) == expected


def test_paths_drop_the_extension():
    assert tokenize_path("src/auth/session.py") == ["src", "auth", "session"]
    assert tokenize_path("src/auth/session_manager.py") == [
        "src",
        "auth",
        "session",
        "manager",
        "session_manager",
    ]


def test_string_literals_split_like_prose():
    assert tokenize_code('retry = headers["retry-after"]') == ["retry", "headers", "retry", "after"]


def test_questions_tokenize_like_code():
    tokens = tokenize_code("Where is parseHTTPHeader called?")
    assert {"parsehttpheader", "parse", "http", "header"} <= set(tokens)


def test_lexical_text_is_space_joined():
    assert lexical_text("MAX_RETRY_COUNT = 3") == "max retry count max_retry_count 3"
