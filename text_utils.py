import ast

import numpy as np
from transformers import PreTrainedTokenizerBase


def parse_conversation(cell: str) -> list[str]:
    """
    Parses a stringified list of conversation turns.

    Args:
        cell: A raw CSV cell expected to look like a Python list
        literal (e.g., '["turn1", "turn2"]').
    Returns:
        The parsed list of turns (e.g., ["turn1", "turn2"]). If parse
        fails, it returns [cell], where cell is the raw string.
    """
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError, TypeError):
        return [cell]

def format_conversation(turns: list[str]) -> str:
    """
    Joins conversation turns into one string for tokenization.

    Args:
        turns: A list of conversation turns to be joined.
    Returns:
        The turns joined with a blank line between each. Any leftover
        JSON-style escaping slashes cleaned up (backslash-escaped
        slashes, unpaired surrogate characters).
    """
    text = "\n\n".join(turns).replace("\\/", "/")
    return text.encode("utf", errors="ignore").decode("utf-8")

def token_lengths(
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase
) -> list[int]:
    """
    Computes how many tokens are in each text.

    Args:
        texts: A list of texts.
        tokenizer: The tokenizer to be used for counting.
    Returns:
        A list of the number of tokens per text.
    """
    encoded = tokenizer(texts, add_special_tokens=False)
    return [len(ids) for ids in encoded["input_ids"]]

def summarize(name: str, lengths: list[int]) -> None:
    """
    Prints the summary of the token counts for the given category.

    Args:
        name: Name of category.
        lengths: A list of the number of tokens for the category.
    """
    arr = np.array(lengths)
    print(
        f"{name}: mean={arr.mean():.1f}, median={np.median(arr):.0f}, "
        f"95th={np.percentile(arr, 95):.0f}, "
        f"99th={np.percentile(arr, 99):.0f}, max={arr.max()}"
    )