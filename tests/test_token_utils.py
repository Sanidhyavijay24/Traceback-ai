"""
Tests for token counting utilities.
"""

from tracebackai.token_utils import count_tokens


def test_count_tokens_none():
    """Verify count_tokens returns None when given None arguments."""
    assert count_tokens(None) is None
    assert count_tokens(None, None) is None


def test_count_tokens_strings():
    """Verify token counts for standard strings."""
    tokens = count_tokens("Hello, world!")
    assert tokens is not None
    assert tokens > 0


def test_count_tokens_multiple_inputs():
    """Verify token counts across multiple inputs/outputs."""
    t1 = count_tokens("hello")
    t2 = count_tokens("world")
    combined = count_tokens("hello", "world")
    assert combined is not None
    assert combined >= (t1 or 0)


def test_count_tokens_objects():
    """Verify token counting on dictionaries and lists."""
    data = {"key": "value", "list": [1, 2, 3]}
    tokens = count_tokens(data)
    assert tokens is not None
    assert tokens > 0
