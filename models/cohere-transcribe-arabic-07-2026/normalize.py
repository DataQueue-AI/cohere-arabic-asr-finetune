"""Text cleaning / normalization for Arabic multi-dialect ASR.

Design goals for the `oddadmix/dialectal-arabic-lahgtna-v2-*` data:

  * Strip tashkil (harakat), dagger-alef and tatweel  -> the target is
    undiacritized text.
  * Remove non-verbal annotation tags like ``[laughter]``, ``[exhale]``,
    ``[inhale]``, ``[mumble]`` etc. (anything in square brackets).
  * Collapse whitespace.

Deliberately *kept*, because this is a multi-dialect corpus and these carry
real information:

  * Dialectal consonants  گ ڨ چ پ ژ ڥ ...  (Maghrebi / Iraqi / Gulf spelling).
  * Sentence punctuation  ، ؟ ! . ...  (Whisper can and should predict it).
  * Latin letters / digits that appear inside transcripts.

Aggressive letter folding (أإآ->ا, ى->ي, ة->ه) is available via
``normalize_letters=True`` but is OFF by default — for a general multi-dialect
model it tends to hurt real-world output. If you enable it, enable it for BOTH
training targets and evaluation references so WER stays consistent.
"""

from __future__ import annotations

import re
import unicodedata

# --- Arabic diacritics / marks to remove -----------------------------------
# Harakat + tanwin + shadda + sukun + maddah + combining hamza (064B-0652),
# extra combining marks (0653-065F), superscript "dagger" alef (0670),
# Arabic small high/low Quranic marks (06D6-06ED), and honorific marks
# (0610-061A). Tatweel/kashida (0640) is elongation-only and also dropped.
_DIACRITICS = (
    "ؐ-ؚ"   # Arabic honorific / small high marks
    "ً-ٟ"   # harakat, tanwin, shadda, sukun, combining hamza, ...
    "ٰ"          # superscript (dagger) alef
    "ۖ-ۜ"   # small high Quranic marks
    "۟-ۨ"
    "۪-ۭ"
)
_DIACRITICS_RE = re.compile(f"[{_DIACRITICS}]")
_TATWEEL_RE = re.compile("ـ+")

# Bracketed non-verbal tags: [laughter], [exhale], [ mumble ], [noise], ...
_BRACKET_TAG_RE = re.compile(r"\[[^\]]*\]")
# Occasionally annotators use <...> or (( )) style tags; strip those too.
_ANGLE_TAG_RE = re.compile(r"<[^>]*>")

# Zero-width / bidi control characters that sometimes ride along in Arabic text.
_ZERO_WIDTH_RE = re.compile("[​-‏‪-‮⁦-⁩﻿]")

_WS_RE = re.compile(r"\s+")

# --- Optional letter folding (OFF by default) ------------------------------
_LETTER_FOLD = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي",
    "ؤ": "و",
    "ئ": "ي",
    "ة": "ه",
    "گ": "ك", "ك": "ك",
}


def remove_diacritics(text: str) -> str:
    text = _DIACRITICS_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)
    return text


def remove_nonverbal_tags(text: str) -> str:
    text = _BRACKET_TAG_RE.sub(" ", text)
    text = _ANGLE_TAG_RE.sub(" ", text)
    return text


def clean_text(text: str, normalize_letters: bool = False) -> str:
    """Full cleaning pipeline. Returns '' if nothing usable remains."""
    if not text:
        return ""
    # Canonical form first, so precomposed vs. decomposed diacritics behave.
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = remove_nonverbal_tags(text)
    text = remove_diacritics(text)
    if normalize_letters:
        text = "".join(_LETTER_FOLD.get(ch, ch) for ch in text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# --- self test -------------------------------------------------------------
_SAMPLES = [
    "طريقة زوجي، كي رجعت ليه ... عايشين [inhale]",
    "مَدارِسْ تقريبًا، وْمُستشفىٰ وَاحِدة فقط [exhale] وكانت الخدمات.",
    "شُو فُهُّم عِند الجَمَال؟ ... طلبية كبيرة؟ [laughter]",
    "هذيك الشمس خرجت وماذا بيا [mumble] [exhale]",
    "المسؤول اللجنة انه يخليه، گال له اني محتاج هذا الراتب",  # keep گ
    "و كانْ مانْقْدروشْ ... ما عَنْدي حاجة. ڨالي و كانْ [exhale]",  # keep ڨ
    "[laughter]",          # -> becomes empty
    "فهَدول الإشياء  الكتير   مهمين",  # multiple spaces
]

if __name__ == "__main__":
    import sys

    fold = "--fold" in sys.argv
    print(f"clean_text(normalize_letters={fold})\n" + "=" * 60)
    for s in _SAMPLES:
        out = clean_text(s, normalize_letters=fold)
        print(f"IN : {s}")
        print(f"OUT: {out!r}")
        print("-" * 60)
