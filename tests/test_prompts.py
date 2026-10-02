"""Prompt templates and the excerpt format."""

import pytest

from cqa.errors import PromptNotFoundError
from cqa.generate.prompts import LOW_CONFIDENCE_NOTICE, format_excerpt, load_template, prompt_hash, render
from cqa.types import Context

from helpers import chunk

VALIDATE = chunk(
    7,
    "src/auth/session.py",
    88,
    90,
    kind="method",
    symbol="SessionManager.validate_token",
    text='def validate_token(self, token: str) -> User:\n    """Return the user."""\n    if not token:',
)


def test_excerpt_header_and_numbered_lines():
    lines = format_excerpt("C1", VALIDATE).splitlines()
    assert lines[0] == "[C1] src/auth/session.py L88-90 (method SessionManager.validate_token)"
    assert lines[1] == "<code>"
    assert lines[2].strip().startswith("88 | def validate_token")
    assert lines[4].strip().startswith("90 |")
    assert lines[-1] == "</code>"


def test_a_skeleton_excerpt_names_its_citable_lines():
    skeleton = chunk(
        9,
        "src/auth/session.py",
        27,
        136,
        kind="class_skeleton",
        citable=(27, 30),
        symbol="SessionManager",
        text="class SessionManager:\n    ...",
    )
    excerpt = format_excerpt("C2", skeleton)
    assert "L27-30" in excerpt.splitlines()[0]
    assert " | " not in excerpt


@pytest.mark.parametrize("version", ["minimal-v0"])
def test_every_field_is_filled(version):
    ctx = Context(chunks=[VALIDATE], token_count=40, low_confidence=False)
    prompt = render(version, "toy", "abc123", ctx, "Where is the session token validated?")
    for field in ("{repo}", "{sha}", "{excerpts}", "{question}", "{low_confidence_notice}"):
        assert field not in prompt
    assert "[C1] src/auth/session.py L88-90" in prompt and "abc123" in prompt
    assert LOW_CONFIDENCE_NOTICE not in prompt


def test_the_notice_appears_only_at_low_confidence():
    ctx = Context(chunks=[VALIDATE], token_count=40, low_confidence=True)
    assert LOW_CONFIDENCE_NOTICE in render("minimal-v0", "toy", "abc123", ctx, "q")


def test_braces_in_code_survive():
    code = chunk(8, "src/auth/session.py", 53, 53, text='        claims = {"sid": session.id}')
    ctx = Context(chunks=[code], token_count=10, low_confidence=False)
    assert 'claims = {"sid": session.id}' in render("minimal-v0", "toy", "abc123", ctx, "q")


def test_the_agent_prefix_relabels_excerpts():
    ctx = Context(chunks=[VALIDATE], token_count=40, low_confidence=False)
    assert "[E1] src/auth/session.py" in render("minimal-v0", "toy", "abc123", ctx, "q", label_prefix="E")


def test_an_unknown_version_is_an_error():
    with pytest.raises(PromptNotFoundError):
        load_template("grounded-v99")


def test_prompt_hash_is_sha256_hex():
    assert prompt_hash("a") == "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
