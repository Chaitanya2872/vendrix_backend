"""Token patterns for Bill of Entry text, and the reasoning behind them.

**Why this document is parsed by pattern, not by label adjacency.**

The invoice parsers work by finding a label and reading the value beside it.
That works because an invoice is laid out as reading text. A Bill of Entry is
a dense ruled grid, and OCR emits it in detection order, not reading order --
the label of one column routinely lands between the value of another column
and its own. Measured on a real ICEGATE print, the Part-II value block came
out as:

    3.DESCRIPTION
    36393.12            <- 14.ASS. VALUE, four labels away from its header
    4.UNIT PRICE
    1                   <- 1.S NO.
    82090090            <- 2.CTH
    WG-4125A,XSYTIN-1,INSERT
    5.QUANTITY 6.UQC
    7.AMOUNT
    11.940000           <- 4.UNIT PRICE
    PRECISION           <- continuation of 3.DESCRIPTION
    32.000000 NOS       <- 5.QUANTITY 6.UQC
    382.08              <- 7.AMOUNT

No label-adjacency rule recovers that. What *does* survive is that each field
has a distinctive shape: a CTH is eight digits, a quantity carries six
decimal places, an amount carries two, a UQC is a short alphabetic code. So
fields are recovered by shape, ordered by their position in the token stream,
and then cross-checked arithmetically -- ``unit_price * quantity == amount``
is what proves the streams were zipped together correctly. Recovery by
pattern, verification by arithmetic.

**OCR damage this has to absorb.** Observed on the sample, all at high
recogniser confidence:

  * the slash in the "2/7" invoice marker is dropped, giving "27";
  * ``I``/``1`` and ``O``/``0`` swap inside identifiers;
  * a digit is occasionally lost entirely (``6580007`` read as ``658007``),
    which is why nothing here treats a length match as proof of a good read.
"""
from __future__ import annotations

import re

# --- structural anchors ----------------------------------------------------

# The Part-II banner, which both identifies the page and numbers the invoice
# it carries. The slash is optional because OCR drops it: "(Invoice 27)" is
# how "2/7" comes back about a third of the time on a photocopied print.
# Both digit groups are single characters for the same reason a BoE never
# carries more than nine invoices per filing in this format.
PART_TWO_BANNER = re.compile(
    r"PART\s*-?\s*I{2}\b.{0,80}?INVO?I?CE\s*[\s(]\s*(\d)\s*/?\s*(\d)\s*\)?",
    re.IGNORECASE | re.DOTALL,
)

PART_ONE_BANNER = re.compile(r"PART\s*-?\s*I\b\s*-?\s*BILL\s*OF\s*ENTRY", re.IGNORECASE)

# The page footer every ICEGATE print carries. Used to split a flat OCR dump
# back into pages, since text extraction joins pages without a marker.
PAGE_FOOTER = re.compile(r"\bPage\s+(\d{1,3})\s*Of\s*(\d{1,3})\b", re.IGNORECASE)

# Document-identifying phrases. Any one of them is enough for can_parse().
BOE_MARKERS = (
    re.compile(r"BILL\s*OF\s*ENTRY", re.IGNORECASE),
    re.compile(r"INDIAN\s*CUSTOMS", re.IGNORECASE),
    re.compile(r"ICEGATE", re.IGNORECASE),
)

# --- identifiers -----------------------------------------------------------

# A BE number is seven digits. Bounded on both sides so it cannot be cut out
# of a longer run such as an AWB number.
BE_NUMBER = re.compile(r"(?<!\d)(\d{7})(?!\d)")

# Indian port codes: IN + four alphanumerics, e.g. INURG6 (SEZ Hyderabad).
PORT_CODE = re.compile(r"\bIN[A-Z0-9]{4}\b")

PAN = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")
# IEC is the importer PAN plus a branch suffix: ABBCS5682H/1.
IEC_WITH_BRANCH = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\s*/\s*(\d{1,2})\b")
GSTIN = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b")
# Customs broker licence, e.g. AAACZ6666LCH001 -- a PAN, then CH, then three
# digits. The trailing group is the customs house serial.
CB_CODE = re.compile(r"\b([A-Z]{5}\d{4}[A-Z][A-Z]{2}\d{3})\b")

# Customs Tariff Head: eight digits, the key to every goods line.
CTH = re.compile(r"(?<!\d)(\d{8})(?!\d)")

# --- quantities and money --------------------------------------------------

# ICEGATE prints quantity and unit price to six decimals, and amounts to two.
# That difference is the only reliable way to tell a unit price from a line
# amount when they arrive in separate token runs.
SIX_DECIMAL = re.compile(r"(?<![\d.])(\d+\.\d{6})(?![\d])")
TWO_DECIMAL = re.compile(r"(?<![\d.])(\d[\d,]*\.\d{2})(?![\d])")
# Quantity and its unit quantity code, when OCR keeps them on one line.
QUANTITY_WITH_UQC = re.compile(r"(?<![\d.])(\d+\.\d{6})\s+([A-Z]{2,4})\b")

# A bare integer or decimal amount, for duty fields printed without decimals
# (IGST 1104620) alongside ones printed with them (BCD 552862.7).
LOOSE_AMOUNT = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d{1,6})?)(?![\d])")

EXCHANGE_RATE = re.compile(
    r"1\s*([A-Z]{3})\s*=?\s*([\d.]+)\s*INR", re.IGNORECASE
)

# --- dates -----------------------------------------------------------------

DATE_SLASH = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
# 31-AUG-26 / 04-SEP-2026
DATE_DMY_ALPHA = re.compile(r"\b(\d{1,2}-[A-Za-z]{3}-\d{2,4})\b")
ANY_DATE = re.compile(
    r"\b(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-[A-Za-z]{3}-\d{2,4})\b"
)

# --- descriptive text ------------------------------------------------------

# A goods-description fragment: mostly capitals, may carry the part-number
# punctuation ICEGATE uses, and is not a pure number or a field label. Labels
# are excluded by the numbered-prefix test in the parser rather than here,
# because "1.S NO." and "WG-4125A" are not distinguishable by shape alone.
DESCRIPTION_FRAGMENT = re.compile(r"^[A-Z0-9][A-Z0-9 ,.\-/&()]{3,}$")

# Field labels in the ICEGATE grid are numbered: "1.BCD", "14.ASS. VALUE".
# Anything matching this is chrome, never a value.
NUMBERED_LABEL = re.compile(r"^\d{1,2}\s*[.:]\s*[A-Za-z]")

UQC_CODES = frozenset({
    "NOS", "PCS", "KGS", "MTR", "LTR", "SQM", "CBM", "TON", "GMS", "SET",
    "PRS", "BOX", "UNT", "THD", "DOZ",
})

# Country names that appear in the origin/consignment cells. Kept small and
# explicit: the cell is free text, and a general gazetteer would match the
# supplier address instead.
COUNTRY_HINTS = (
    "UNITED STATES", "UNITED KINGDOM", "GERMANY", "FRANCE", "ITALY", "JAPAN",
    "CHINA", "SINGAPORE", "SWITZERLAND", "CANADA", "SPAIN", "NETHERLANDS",
    "BELGIUM", "SWEDEN", "AUSTRIA", "SOUTH KOREA", "TAIWAN", "MEXICO",
    "POLAND", "CZECH REPUBLIC", "TURKEY", "BRAZIL", "AUSTRALIA", "INDIA",
)

INCOTERMS = frozenset({"CIF", "FOB", "CFR", "EXW", "DAP", "DDP", "FCA", "CIP", "CPT"})

CURRENCIES = frozenset({"USD", "EUR", "GBP", "JPY", "CHF", "SGD", "INR", "AUD", "CAD"})


def is_label(line: str) -> bool:
    """Is this token a field label rather than a value?

    Cheap and deliberately conservative -- a value wrongly discarded as a
    label is unrecoverable, so this only rejects the two unambiguous cases:
    the numbered-label convention, and lines that are entirely punctuation.
    """
    stripped = line.strip()
    if not stripped:
        return True
    if NUMBERED_LABEL.match(stripped):
        return True
    return not any(char.isalnum() for char in stripped)
