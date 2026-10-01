"""Tests for text_utils.py"""


from text_utils import format_conversation, parse_conversation


def test_parse_conversation_valid_list():
    assert parse_conversation('["hello", "world"]') == ["hello", "world"]


def test_parse_conversation_falls_back_on_bad_input():
    assert parse_conversation("hello world") == ["hello world"]


def test_format_conversation_joins_with_blank_line():
    assert format_conversation(["hello", "world"]) == "hello\n\nworld"


def test_format_conversation_unescapes_slashes():
    # "\\/" in source is the literal two character \/ (JSON-escape).
    assert format_conversation(["hello\\/world"]) == "hello/world"