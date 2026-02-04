"""
Arabic text processing utilities.
"""

import re
import difflib
from typing import Dict

from .config import GLOSSARY

# Pattern to match Arabic characters
_AR_RE = re.compile(r'[\u0600-\u06FF]')


def arabic_ratio(text: str) -> float:
    """Calculate the ratio of Arabic letters among alphabetic characters."""
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    ar_letters = sum(1 for c in letters if _AR_RE.match(c))
    return ar_letters / max(1, len(letters))


def is_good_arabic_segment(text: str, min_len: int = 20) -> bool:
    """
    Heuristic filter to prevent drift:
    - Must be mostly Arabic.
    - Must not contain common boilerplate hallucinations.
    - Must be long enough (tunable).
    """
    if not text:
        return False
    t = text.strip()
    if len(t) < min_len:
        return False

    # Common boilerplate / subtitle hallucinations (Arabic + English)
    if re.search(
        r'(نانسي|ترجمة|اشتركوا|اشترك|تابعونا|حقوق|محفوظة|موسيقى|قناة|'
        r'Subtitle|Translated|Amara|MBC|Copyright|Rights|Reserved|Music|Nancy|Nana)',
        t, re.IGNORECASE
    ):
        return False

    return arabic_ratio(t) >= 0.75


def glossary_block() -> str:
    """Embed glossary as instructions for LLM prompts."""
    lines = [f"- {ar} => {en}" for ar, en in GLOSSARY.items()]
    return "GLOSSARY (must follow exactly):\n" + "\n".join(lines)


def similarity(a: str, b: str) -> float:
    """Calculate sequence similarity between two strings."""
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def protect_terms(text: str) -> tuple:
    """Replace glossary terms with placeholders for safe translation."""
    mapping = {}
    out = text
    for i, (ar, en) in enumerate(GLOSSARY.items()):
        if ar in out:
            key = f"§§TERM{i}§§"
            mapping[key] = en
            out = out.replace(ar, key)
    return out, mapping


def restore_terms(text: str, mapping: Dict[str, str]) -> str:
    """Restore placeholders back to translated terms."""
    out = text
    for key, val in mapping.items():
        out = out.replace(key, val)
    return out
