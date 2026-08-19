"""Turn a document line into features for the field classifier.

Three views of each line are combined, because each carries signal the others
miss:

  character n-grams   survive OCR damage — "AmountPayabIe" still shares most
                      of its 3-grams with "Amount Payable", where a word-level
                      match would score zero
  word n-grams over   captures the label, which is frequently not on the value
  (previous + line)   line at all: OCR of a two-column totals block emits
                      "Total:" and "1,92,407.04" as separate lines
  shape features      say what a line *is* rather than what it says — a line
                      that is nothing but a money token, a line holding a
                      15-character alphanumeric, where the line sits on the
                      page. These transfer across wordings the training corpus
                      never contained.

Deliberately *not* included: per-field keyword lists. Those would make the
model a regex wearing a classifier's coat, and would cap it at the vocabulary
someone thought of at build time. The lexical signal comes from the n-gram
views, which learn wordings from data instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from .corpus import DATE_TOKEN, GSTIN_TOKEN, MONEY_TOKEN

# Separator between the previous line and the current one in the contextual
# word view. A token that cannot occur in document text keeps the two sides
# from forming spurious cross-boundary bigrams.
CONTEXT_SEPARATOR = " __eol__ "

_LONG_ALNUM = re.compile(r"\b(?=[A-Z0-9/\-]{5,})(?=[^\s]*\d)[A-Z0-9][A-Z0-9/\-]{4,}\b")
_PERCENT = re.compile(r"\d\s*%")
_CURRENCY = re.compile(r"₹|Rs\.?|INR|USD|\$", re.IGNORECASE)


@dataclass
class LineContext:
    """A line plus the neighbourhood the model is allowed to see. Kept to
    immediate neighbours and position: a wider window would let the model key
    off document-level quirks of the synthetic corpus instead of the local
    evidence that generalises."""

    text: str
    previous_text: str = ""
    next_text: str = ""
    line_index: int = 0
    line_count: int = 1


def contexts_from_lines(lines: list[str]) -> list[LineContext]:
    """Build contexts for a whole document — the inference-side counterpart of
    what corpus.py records at training time. Both must agree, or the model
    sees a different neighbourhood than it was trained on."""
    cleaned = [line.strip() for line in lines if line.strip()]
    return [
        LineContext(
            text=line,
            previous_text=cleaned[index - 1] if index > 0 else "",
            next_text=cleaned[index + 1] if index + 1 < len(cleaned) else "",
            line_index=index,
            line_count=len(cleaned),
        )
        for index, line in enumerate(cleaned)
    ]


class SelectText(BaseEstimator, TransformerMixin):
    """Pull one text view out of a list of LineContext."""

    def __init__(self, view: str = "text"):
        self.view = view

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        if self.view == "context":
            return [f"{item.previous_text}{CONTEXT_SEPARATOR}{item.text}" for item in X]
        return [getattr(item, self.view) for item in X]


class ShapeFeatures(BaseEstimator, TransformerMixin):
    """What kind of line is this, regardless of the words in it."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        return [self._describe(item) for item in X]

    @staticmethod
    def _describe(item: LineContext) -> dict[str, float]:
        text = item.text
        characters = len(text) or 1
        tokens = text.split()
        digits = sum(character.isdigit() for character in text)
        alpha = sum(character.isalpha() for character in text)
        upper = sum(character.isupper() for character in text)

        money_tokens = MONEY_TOKEN.findall(text)
        date_tokens = DATE_TOKEN.findall(text)
        money_only = bool(money_tokens) and len(_CURRENCY.sub("", text).replace(" ", "")) <= len(
            max(money_tokens, key=len).replace(" ", "")
        ) + 2

        return {
            # Where on the page. Header fields cluster at the top, totals at
            # the bottom, and that ordering holds across every format.
            "relative_position": item.line_index / max(1, item.line_count - 1),
            "is_first_lines": float(item.line_index < 4),
            "is_last_lines": float(item.line_index >= item.line_count - 6),
            # Composition.
            "length": min(len(text), 200) / 200,
            "token_count": min(len(tokens), 25) / 25,
            "digit_ratio": digits / characters,
            "alpha_ratio": alpha / characters,
            "upper_ratio": upper / max(1, alpha),
            # Value shapes. These are what let the model tell a total from a
            # date from a GSTIN without reading a single word.
            "money_count": min(len(money_tokens), 5) / 5,
            "has_money": float(bool(money_tokens)),
            "is_money_only": float(money_only),
            "has_date": float(bool(date_tokens)),
            "has_gstin_shape": float(bool(GSTIN_TOKEN.search(text.upper()))),
            "has_long_alnum": float(bool(_LONG_ALNUM.search(text.upper()))),
            "has_percent": float(bool(_PERCENT.search(text))),
            "has_currency": float(bool(_CURRENCY.search(text))),
            "has_colon": float(":" in text),
            # The neighbourhood. A bare value line is only interpretable
            # through the line above it.
            "previous_has_colon": float(":" in item.previous_text),
            "previous_is_short": float(0 < len(item.previous_text) <= 30),
            "previous_has_money": float(bool(MONEY_TOKEN.search(item.previous_text))),
            "next_has_money": float(bool(MONEY_TOKEN.search(item.next_text))),
            "next_is_money_only": float(
                bool(MONEY_TOKEN.search(item.next_text)) and len(item.next_text) <= 20
            ),
        }


def build_featureuniser(
    char_max_features: int = 60_000,
    word_max_features: int = 40_000,
    min_df: int = 2,
) -> FeatureUnion:
    """The three views, concatenated into one sparse matrix.

    `sublinear_tf` matters here: table rows repeat tokens (a currency marker
    per column) and raw counts would let a wide row shout down a short one
    carrying the same evidence.
    """
    return FeatureUnion(
        [
            (
                "line_characters",
                Pipeline([
                    ("select", SelectText("text")),
                    ("tfidf", TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 5),
                        min_df=min_df,
                        max_features=char_max_features,
                        sublinear_tf=True,
                        lowercase=True,
                    )),
                ]),
            ),
            (
                "context_words",
                Pipeline([
                    ("select", SelectText("context")),
                    ("tfidf", TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=min_df,
                        max_features=word_max_features,
                        sublinear_tf=True,
                        lowercase=True,
                        token_pattern=r"(?u)\b\w[\w./#-]*\b",
                    )),
                ]),
            ),
            (
                "shape",
                Pipeline([
                    ("select", ShapeFeatures()),
                    ("vectorise", DictVectorizer(sparse=True)),
                    # Sparse-safe scaling: the shape block must not dominate
                    # the L2-normalised tf-idf blocks purely through scale.
                    ("scale", MaxAbsScaler()),
                ]),
            ),
        ],
        transformer_weights={"line_characters": 1.0, "context_words": 1.0, "shape": 0.6},
    )
