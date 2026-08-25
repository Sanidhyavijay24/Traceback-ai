"""
Traceback AI - Token Utilities.

Provides token estimation using tiktoken with fallback.
"""

from typing import Any, Optional

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENCODER = None


def count_tokens(*items: Any) -> Optional[int]:
    """
    Count estimated tokens across all provided items.

    Uses cl100k_base tiktoken encoding if available, falling back to
    length // 4 characters.
    """
    total = 0
    valid_item_found = False

    for item in items:
        if item is None:
            continue
        valid_item_found = True
        text = str(item)
        if not text:
            continue

        if _ENCODER is not None:
            try:
                total += len(_ENCODER.encode(text, disallowed_special=()))
            except Exception:
                total += max(1, len(text) // 4)
        else:
            total += max(1, len(text) // 4)

    return total if valid_item_found else None
