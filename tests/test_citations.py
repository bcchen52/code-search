"""Citation parsing: every grammar form, malformed tokens, and sentence attachment."""

from cqa.generate.citations import find_malformed, parse_citations, split_sentences


def labels(answer: str):
    return [(c.label, c.lines, c.sentence) for c in parse_citations(answer)]


def test_plain_and_narrowed():
    answer = "Tokens are checked in `validate_token` [C1:L95-105]. Expired sessions are revoked [C1][C2]."
    assert labels(answer) == [("C1", (95, 105), 0), ("C1", None, 1), ("C2", None, 1)]


def test_a_citation_after_the_full_stop_belongs_to_the_sentence_before():
    assert labels("The limit is five sessions. [C2] Older ones are revoked [C3].") == [
        ("C2", None, 0),
        ("C3", None, 1),
    ]


def test_mid_sentence_citations():
    assert [c.sentence for c in parse_citations("`refresh` [C1] calls `validate_token` [C2] first.")] == [
        0,
        0,
    ]


def test_dots_inside_names_do_not_end_sentences():
    answer = (
        "It reads config/settings.yaml [C4] and calls auth.session.validate [C2]. Nothing else does [C3]."
    )
    assert len(split_sentences(answer)) == 2
    assert [c.sentence for c in parse_citations(answer)] == [0, 0, 1]


def test_offsets_point_at_the_token():
    answer = "See [C12:L3-9]."
    (c,) = parse_citations(answer)
    assert answer[c.start : c.end] == "[C12:L3-9]"


def test_agent_labels_use_the_same_grammar():
    assert labels("It is read in `load` [E4:L30-41].") == [("E4", (30, 41), 0)]


def test_malformed_tokens_are_not_citations():
    answer = "Bad forms: [C] [c2] [C3:95-103] [C4:L10] [C5:L10-]. A real one [C6]."
    assert labels(answer) == [("C6", None, 1)]
    assert find_malformed(answer) == ["[C]", "[c2]", "[C3:95-103]", "[C4:L10]", "[C5:L10-]"]


def test_markdown_brackets_are_not_attempts():
    assert find_malformed("See [Config](docs/config.md) and the [CHANGELOG].") == []
