"""
Shared utilities.
Kept: token counting utility.
Everything else moved to dedicated modules in v2.0:
  - Search: search/ package
  - PDF/Output: output_compiler.py
  - Embeddings: embeddings.py
"""

import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: The text to count tokens for.
        model: Model name for tokenizer selection.

    Returns:
        Number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def truncate_to_tokens(text: str, max_tokens: int = 4000, model: str = "gpt-4") -> str:
    """
    Truncate text to a maximum number of tokens.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens.
        model: Model name for tokenizer selection.

    Returns:
        Truncated text.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
