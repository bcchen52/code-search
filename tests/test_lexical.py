"""The per-index full-text table."""

from dataclasses import replace

import pytest

from cqa.chunking.subtokenize import lexical_text
from cqa.index.lexical import LexicalIndex
from cqa.retrieve.bm25 import to_match

from helpers import chunk


def indexed(conn, index_id, texts):
    lex = LexicalIndex(conn, index_id)
    lex.create()
    lex.add(
        replace(chunk(i, "a.py", i, i, text=t), lexical_text=lexical_text(t)) for i, t in enumerate(texts, 1)
    )
    return lex


def test_the_table_name_requires_a_hex_index_id(main_db):
    assert LexicalIndex(main_db, "ab12cd34").table == "fts_ab12cd34"
    with pytest.raises(ValueError):
        _ = LexicalIndex(main_db, "x; DROP TABLE chunks").table


def test_whole_identifiers_stay_single_tokens(main_db):
    lex = indexed(main_db, "ab12", ["MAX_RETRY_COUNT = 3", "Set the max retry count in settings."])
    assert [cid for cid, _ in lex.search(to_match(["max_retry_count"]), k=10)] == [1]


def test_scores_are_higher_for_better_matches(main_db):
    lex = indexed(main_db, "cd34", ["validate the token", "validate token token token", "parse config"])
    hits = lex.search(to_match(["token"]), k=10)
    assert [cid for cid, _ in hits] == [2, 1]
    assert hits[0][1] > hits[1][1]


def test_indexes_do_not_share_statistics(main_db):
    a = indexed(main_db, "aa11", ["validate token", "parse config"])
    before = a.search(to_match(["token"]), k=10)
    indexed(main_db, "bb22", ["token " * 50] * 20)
    assert a.search(to_match(["token"]), k=10) == before


def test_to_match_quotes_every_token():
    assert to_match(["validate", "token"]) == '"validate" OR "token"'
    assert to_match(['say "hi"']) == '"say ""hi"""'
