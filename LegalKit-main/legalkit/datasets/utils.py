import re


def clean_prediction(text: str) -> str:
    """
    Normalize model raw output to the final answer span.

    Rules (in order):
    - If a closing </think> exists, keep the content after the last </think>
      (filters out chain-of-thought blocks).
    - Else, if the substring "think" appears anywhere (case-insensitive),
      drop everything before the last occurrence of "think" and keep the rest.
    - Trim leading junk symbols after truncation.
    """
    if text is None:
        return ""

    s = str(text)

    # Standard CoT tag: keep after the last closing tag.
    if "</think>" in s:
        s = s.split("</think>")[-1]
    else:
        # Fallback: any occurrence of "think" implies we should keep only the tail.
        lower = s.lower()
        idx = lower.rfind("think")
        if idx != -1:
            s = s[idx + len("think") :]

    s = s.strip()

    # If there are leftover <think>...</think> blocks (non-standard nesting), remove them.
    s = re.sub(r"(?is)<think>.*?</think>", "", s).strip()

    # Drop leading non-content symbols (e.g., '>', ':', '-', whitespace).
    s = re.sub(r"^[^\w\u4e00-\u9fffA-Za-z]+", "", s).strip()

    return s
