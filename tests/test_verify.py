"""Mechanical citation checks."""

import pytest

from cqa.generate.citations import parse_citations
from cqa.generate.verify import verify

from helpers import chunk

LABELS = {
    "C1": chunk(7, "src/auth/session.py", 88, 121, kind="method", symbol="SessionManager.validate_token"),
    "C2": chunk(
        9, "src/auth/session.py", 27, 136, kind="class_skeleton", citable=(27, 30), symbol="SessionManager"
    ),
}


def check(answer: str):
    return verify(answer, parse_citations(answer), LABELS)


def statuses(answer: str) -> list[str]:
    return [c.status for c in check(answer).checks]


def test_an_unknown_label_is_fabricated():
    assert statuses("It happens in `validate_token` [C9].") == ["fabricated"]


def test_a_narrowed_range_must_sit_inside_the_chunk():
    assert statuses("Expiry is checked here [C1:L107-110].") == ["valid"]
    assert statuses("Expiry is checked here [C1:L80-95].") == ["out_of_range"]


def test_a_skeleton_accepts_its_header_lines_only():
    assert statuses("`SessionManager` signs with HS256 [C2:L30-30].") == ["valid"]
    assert statuses("`SessionManager` validates tokens [C2:L95-100].") == ["out_of_range"]
    assert statuses("`SessionManager` holds the store [C2].") == ["valid"]


def test_a_sentence_that_names_code_needs_a_citation():
    assert check("The check lives in `validate_token`. It raises on expiry [C1].").uncited_sentences == [0]


def test_a_plain_sentence_needs_no_citation():
    assert check("I could not find OAuth handling in these excerpts.").uncited_sentences == []


def test_validity_is_the_share_of_valid_citations():
    assert check("A [C1]. B [C9]. C [C1:L90-91].").validity == pytest.approx(2 / 3)
    assert check("No citations here.").validity is None


def test_checks_resolve_spans_for_the_ui():
    narrowed, plain = check("See `validate_token` [C1:L95-105]. See also [C1].").checks
    assert (narrowed.path, narrowed.lines) == ("src/auth/session.py", (95, 105))
    assert plain.lines == (88, 121)
