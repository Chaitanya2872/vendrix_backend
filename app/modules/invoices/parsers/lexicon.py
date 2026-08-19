"""The label lexicon: every wording this pipeline knows for every field.

**Keywords are not trained.** PaddleOCR converts pixels to characters and
knows nothing about invoices; the trained model in `app/ml/ocr` classifies
lines by *context and position*, learning wordings nobody listed. This module
is the third layer — a curated list of what vendors actually call things,
matched fuzzily so an OCR slip does not cost a field.

Three properties make this list work where a naive one does not:

**Ordering is significance, not alphabetical.** `taxable value` is checked
before `total` because "Total Taxable Value" contains both, and the more
specific reading is the right one. Reordering these lists changes behaviour.

**Negative labels exist.** `amount in words` and `total quantity` both
contain a money label and must never be read as one. Listing what a field is
*not* is as load-bearing as listing what it is.

**Matching is tolerant.** Comparison happens after normalisation and through
an OCR-confusion map, so `AmountPayabIe` (capital-i for lowercase-L, a
classic recognition error) still matches `Amount Payable`. An exact-match
list is precise on clean text and blind on the scans that need it most.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --- normalisation ---------------------------------------------------------

# Character pairs OCR routinely confuses. Applied to *both* sides of a
# comparison so the mapping never has to guess which direction the error went.
# Everything collapses toward the digit or the simpler glyph.
OCR_CONFUSIONS = str.maketrans({
    "0": "o", "O": "o", "o": "o", "Q": "o", "D": "o",
    "1": "l", "I": "l", "|": "l", "!": "l", "L": "l",
    "5": "s", "S": "s",
    "8": "b", "B": "b",
    "2": "z", "Z": "z",
    "6": "g", "G": "g",
})

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace.

    Accent stripping matters because OCR sprays diacritics onto clean text
    under noise: a stray tilde over an 'n' should not cost a label match.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = _PUNCTUATION.sub(" ", without_accents.lower())
    return _WHITESPACE.sub(" ", cleaned).strip()


def confusion_key(text: str) -> str:
    """Normalised text with OCR-confusable characters collapsed.

    Used only as a fallback when normalised comparison fails: it is lossy
    enough that `10` and `lo` become the same string, so it must never be the
    primary test.
    """
    return normalise(text).translate(OCR_CONFUSIONS)


# --- field taxonomy --------------------------------------------------------


@dataclass(frozen=True)
class FieldLabels:
    """Every wording for one field, plus what disqualifies a match."""

    field: str
    kind: str                                  # money | date | identifier | text | gstin | percent
    labels: tuple[str, ...]
    negative: tuple[str, ...] = ()
    # Lower sorts first when two fields both match a line. Specific fields
    # (taxable value) must outrank general ones (total).
    priority: int = 50
    aliases_are_prefixes: bool = False

    @property
    def normalised_labels(self) -> tuple[str, ...]:
        return tuple(normalise(label) for label in self.labels)

    @property
    def normalised_negatives(self) -> tuple[str, ...]:
        return tuple(normalise(label) for label in self.negative)


# Labels that must never be read as a money field, wherever they appear.
# "Amount in Words" contains "Amount"; "Total Quantity" contains "Total".
MONEY_NEGATIVES: tuple[str, ...] = (
    "amount in words", "rupees only", "in words", "amount chargeable in words",
    "total quantity", "total qty", "total items", "total no of items",
    "total pages", "page", "total invoice count",
)

# --- identity --------------------------------------------------------------

INVOICE_NUMBER = FieldLabels(
    "invoice_number", "identifier", priority=10,
    labels=(
        "tax invoice no", "tax invoice number", "invoice number", "invoice no",
        "invoice #", "invoice nbr", "inv no", "inv #", "invoice id",
        "bill number", "bill no", "document no", "document number",
        "voucher no", "voucher number", "receipt no", "reference no", "ref no",
        "our invoice no", "invoice",
    ),
    negative=("invoice date", "invoice value", "invoice total", "invoice amount"),
)

INVOICE_DATE = FieldLabels(
    "invoice_date", "date", priority=10,
    labels=(
        "invoice date", "date of invoice", "bill date", "date of issue",
        "issue date", "invoice dt", "dated", "date",
    ),
    negative=("due date", "po date", "order date", "delivery date", "date of supply"),
)

DUE_DATE = FieldLabels(
    "due_date", "date", priority=10,
    labels=(
        "due date", "payment due date", "payment due", "due on", "pay by",
        "net due date", "maturity date", "date due",
    ),
)

PO_NUMBER = FieldLabels(
    "purchase_order_number", "identifier", priority=15,
    labels=(
        "purchase order number", "purchase order no", "po number", "po no",
        "p o no", "order no", "order number", "order reference",
        "buyers order no", "your order no", "customer po", "po ref",
    ),
    negative=("po date", "purchase order date"),
)

PO_DATE = FieldLabels(
    "purchase_order_date", "date", priority=15,
    labels=("po date", "purchase order date", "order date", "buyers order date"),
)

DELIVERY_REFERENCE = FieldLabels(
    "delivery_reference", "identifier", priority=20,
    labels=(
        "delivery note", "delivery note no", "challan no", "delivery challan",
        "dc no", "e way bill no", "eway bill no", "ewb no", "lr no",
        "dispatch doc no", "transport doc no", "docket no",
    ),
)

IRN = FieldLabels(
    "irn", "identifier", priority=20,
    labels=("irn", "invoice reference number", "ack no", "acknowledgement no",
            "acknowledgement number", "ack number"),
)

# --- parties ---------------------------------------------------------------

VENDOR_LABELS = FieldLabels(
    "vendor_name", "text", priority=20,
    labels=(
        "sold by", "seller", "supplier", "vendor", "issued by", "billed from",
        "from", "remit to", "service provider", "supplier name", "seller name",
    ),
)

CUSTOMER_LABELS = FieldLabels(
    "customer_name", "text", priority=20,
    labels=(
        "bill to", "billed to", "bill to party", "billing address", "buyer",
        "customer", "customer name", "sold to", "invoice to", "party name",
        "client", "buyer name", "to",
    ),
    negative=("ship to", "shipped to", "consignee", "delivery address"),
)

SHIP_TO_LABELS = FieldLabels(
    "ship_to_name", "text", priority=20,
    labels=(
        "ship to", "shipped to", "shipping address", "consignee",
        "delivery address", "deliver to", "place of delivery", "despatch to",
    ),
)

GSTIN_LABELS = FieldLabels(
    "gstin", "gstin", priority=15,
    labels=("gstin", "gst no", "gst number", "gstin uin", "gst reg no",
            "gst registration no", "gstin no"),
)

TAX_ID_LABELS = FieldLabels(
    "tax_id", "identifier", priority=25,
    labels=("pan", "pan no", "pan number", "cin", "tan", "vat no", "vat reg no",
            "tin", "udyam", "udyam registration", "msme reg no", "iec", "tax id"),
)

CONTACT_LABELS = FieldLabels(
    "contact", "text", priority=40,
    labels=("email", "e mail", "phone", "tel", "telephone", "mobile", "mob",
            "contact no", "contact", "fax", "website", "web"),
)

# --- money -----------------------------------------------------------------
#
# Ordering within this tuple is behaviour. `taxable value` precedes `sub
# total` which precedes `total`, because a line reading "Total Taxable Value"
# matches all three and only the first reading is correct.

TAXABLE_AMOUNT = FieldLabels(
    "taxable_amount", "money", priority=10,
    labels=(
        "total taxable value", "taxable value", "taxable amount",
        "assessable value", "total assessable value", "net taxable value",
    ),
    negative=MONEY_NEGATIVES,
)

SUBTOTAL = FieldLabels(
    "subtotal", "money", priority=20,
    labels=(
        "sub total", "subtotal", "sub-total", "total before tax",
        "amount before tax", "net amount", "gross amount", "basic amount",
        "total before gst", "value of supply",
    ),
    negative=MONEY_NEGATIVES,
)

DISCOUNT = FieldLabels(
    "discount_amount", "money", priority=20,
    labels=(
        "less discount", "trade discount", "cash discount", "discount amount",
        "discount", "less", "rebate",
    ),
    # No "%" variants here: normalisation strips punctuation, so "disc %"
    # becomes "disc" and would disqualify every discount label there is.
    # A rate column is excluded by value typing instead — a percentage
    # does not parse as money.
    negative=MONEY_NEGATIVES,
)

FREIGHT = FieldLabels(
    "freight_amount", "money", priority=25,
    labels=(
        "packing and forwarding", "packing forwarding", "p and f",
        "freight charges", "freight", "shipping charges", "shipping",
        "delivery charges", "transport charges", "insurance charges",
        "handling charges", "other charges", "misc charges", "installation charges",
    ),
    negative=MONEY_NEGATIVES,
)

CGST = FieldLabels(
    "cgst_amount", "money", priority=10,
    labels=("cgst amount", "central gst", "c gst", "cgst"),
    negative=MONEY_NEGATIVES,
)

SGST = FieldLabels(
    "sgst_amount", "money", priority=10,
    labels=("sgst amount", "state gst", "s gst", "sgst", "utgst", "ugst"),
    negative=MONEY_NEGATIVES,
)

IGST = FieldLabels(
    "igst_amount", "money", priority=10,
    labels=("igst amount", "integrated gst", "i gst", "igst"),
    negative=MONEY_NEGATIVES,
)

CESS = FieldLabels(
    "cess_amount", "money", priority=15,
    labels=("gst compensation cess", "compensation cess", "cess amount", "cess"),
    negative=MONEY_NEGATIVES,
)

TAX_TOTAL = FieldLabels(
    "tax_amount", "money", priority=25,
    labels=(
        "total tax amount", "total tax", "tax amount", "total gst",
        "gst amount", "output tax", "vat amount", "service tax", "sales tax",
    ),
    # "tax rate", not "tax %": punctuation does not survive normalisation,
    # and "tax %" reduced to "tax" disqualified every label this field
    # has — silently removing tax_amount from extraction entirely.
    negative=MONEY_NEGATIVES + ("tax rate",),
)

ROUND_OFF = FieldLabels(
    "round_off", "money", priority=15,
    labels=("rounded off", "round off", "roundoff", "rounding", "r o", "adjustment"),
    negative=MONEY_NEGATIVES,
)

TOTAL = FieldLabels(
    "total_amount", "money", priority=30,
    labels=(
        "total amount payable", "total invoice value", "invoice total",
        "grand total", "net payable", "amount payable", "total payable",
        "total amount", "total due", "invoice value", "total value",
        "total incl tax", "total including tax", "total",
    ),
    negative=MONEY_NEGATIVES + (
        "sub total", "subtotal", "total tax", "total taxable value",
        "taxable value", "total quantity", "total discount", "total before tax",
    ),
)

AMOUNT_PAID = FieldLabels(
    "amount_paid", "money", priority=25,
    labels=("amount paid", "payment received", "advance received", "less advance",
            "advance", "paid"),
    negative=MONEY_NEGATIVES,
)

AMOUNT_DUE = FieldLabels(
    "amount_due", "money", priority=25,
    labels=("balance due", "amount due", "outstanding amount", "outstanding",
            "net due", "balance payable"),
    negative=MONEY_NEGATIVES,
)

# --- other header fields ---------------------------------------------------

PAYMENT_TERMS = FieldLabels(
    "payment_terms", "text", priority=35,
    labels=("terms of payment", "payment terms", "credit period", "payment condition"),
)

PLACE_OF_SUPPLY = FieldLabels(
    "place_of_supply", "text", priority=35,
    labels=("place of supply", "state name", "state code", "state"),
)

REVERSE_CHARGE = FieldLabels(
    "reverse_charge", "text", priority=40,
    labels=("whether tax payable on reverse charge", "reverse charge",
            "tax payable on reverse charge"),
)

CURRENCY = FieldLabels(
    "currency", "text", priority=40,
    labels=("currency", "curr", "currency code"),
)

# --- banking ---------------------------------------------------------------

BANK_ACCOUNT = FieldLabels(
    "bank_account_number", "identifier", priority=30,
    labels=("account no", "account number", "a c no", "ac no", "bank account no",
            "beneficiary account"),
)

BANK_IFSC = FieldLabels(
    "bank_ifsc", "identifier", priority=30,
    labels=("ifsc code", "ifsc", "swift code", "swift", "bic", "iban"),
)

BANK_NAME = FieldLabels(
    "bank_name", "text", priority=35,
    labels=("bank name", "bank", "branch name", "branch"),
)

UPI = FieldLabels(
    "upi_id", "identifier", priority=35,
    labels=("upi id", "upi", "vpa", "virtual payment address"),
)


# Every field, ordered so that a more specific label is always tried before a
# more general one that contains it.
ALL_FIELDS: tuple[FieldLabels, ...] = tuple(sorted(
    (
        INVOICE_NUMBER, INVOICE_DATE, DUE_DATE, PO_NUMBER, PO_DATE,
        DELIVERY_REFERENCE, IRN,
        VENDOR_LABELS, CUSTOMER_LABELS, SHIP_TO_LABELS, GSTIN_LABELS,
        TAX_ID_LABELS, CONTACT_LABELS,
        TAXABLE_AMOUNT, SUBTOTAL, DISCOUNT, FREIGHT,
        CGST, SGST, IGST, CESS, TAX_TOTAL, ROUND_OFF, TOTAL,
        AMOUNT_PAID, AMOUNT_DUE,
        PAYMENT_TERMS, PLACE_OF_SUPPLY, REVERSE_CHARGE, CURRENCY,
        BANK_ACCOUNT, BANK_IFSC, BANK_NAME, UPI,
    ),
    key=lambda spec: spec.priority,
))

FIELD_BY_NAME: dict[str, FieldLabels] = {spec.field: spec for spec in ALL_FIELDS}

MONEY_FIELDS: frozenset[str] = frozenset(
    spec.field for spec in ALL_FIELDS if spec.kind == "money"
)


# --- matching --------------------------------------------------------------


@dataclass
class LabelMatch:
    field: str
    label: str
    score: float           # 0..1, how well the text matched the label
    exact: bool
    matched_text: str


def _ratio(left: str, right: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio()


def _similarity(left: str, right: str) -> float:
    """Token-set similarity in [0, 1].

    Implemented here rather than delegated to a fuzzy-matching library on
    purpose. The thresholds in this module are tuned to *this* function, and
    a library that is present in one environment and absent in another would
    silently change what counts as a match — the kind of difference that
    shows up as "extraction is worse in production" with nothing in the diff
    to explain it.

    The algorithm is the standard token-set comparison: score the shared
    tokens against each side's full token list, and take the best. It is what
    makes `Invoice No` match `Invoice No.` and `No Invoice` alike, while
    keeping `Invoice No` well clear of `Invoice Date`.
    """
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0

    shared = " ".join(sorted(left_tokens & right_tokens))
    left_only = " ".join(sorted(left_tokens - right_tokens))
    right_only = " ".join(sorted(right_tokens - left_tokens))

    combined_left = f"{shared} {left_only}".strip()
    combined_right = f"{shared} {right_only}".strip()

    if not shared:
        # No token in common: fall back to a plain character comparison so a
        # single mangled word ("lnvoice") can still match.
        return _ratio(left, right)

    # Only the full-versus-full comparison. The textbook token-set ratio also
    # scores the shared tokens against each side and takes the best, which
    # makes any subset a *perfect* match: "Invoice" would score 1.0 against
    # "Invoice Total", and an invoice number would be extracted as the
    # invoice total. Dropping those two terms costs nothing real — a genuine
    # near-match still has nearly all its tokens in common — and removes a
    # whole class of confidently-wrong field assignments.
    return _ratio(combined_left, combined_right)


def match_label(text: str, spec: FieldLabels, threshold: float = 0.85) -> LabelMatch | None:
    """Does `text` name this field?

    Exact containment first, fuzzy second, OCR-confusion-tolerant third. The
    order matters: fuzzy matching alone would let `Invoice Date` score highly
    against `Invoice No`, and the exact pass is what keeps the common case
    unambiguous.
    """
    candidate = normalise(text)
    if not candidate:
        return None

    # A negative label disqualifies outright, before any positive match is
    # attempted. "Total Quantity" must not reach the money matcher at all.
    for negative in spec.normalised_negatives:
        if negative and negative in candidate:
            return None

    # 1. Exact — the whole text is the label, or begins with it.
    for label, normalised_label in zip(spec.labels, spec.normalised_labels):
        if candidate == normalised_label:
            return LabelMatch(spec.field, label, 1.0, True, text)

    # 2. Containment — the label appears within a longer fragment.
    for label, normalised_label in zip(spec.labels, spec.normalised_labels):
        if normalised_label and normalised_label in candidate:
            # Longer labels are stronger evidence: matching "total amount
            # payable" says more than matching "total".
            coverage = len(normalised_label) / max(len(candidate), 1)
            return LabelMatch(spec.field, label, 0.80 + 0.19 * coverage, False, text)

    # 3. Fuzzy — tolerates a dropped letter or a joined word.
    best: LabelMatch | None = None
    for label, normalised_label in zip(spec.labels, spec.normalised_labels):
        score = _similarity(candidate, normalised_label)
        if score >= threshold and (best is None or score > best.score):
            best = LabelMatch(spec.field, label, score * 0.95, False, text)
    if best is not None:
        return best

    # 4. OCR confusion — the last resort, for text the recogniser mangled.
    confused = confusion_key(text)
    for label, normalised_label in zip(spec.labels, spec.normalised_labels):
        if len(normalised_label) < 5:
            continue  # too short to survive this much collapsing safely
        if confusion_key(normalised_label) in confused:
            return LabelMatch(spec.field, label, 0.75, False, text)

    return None


def match_any(text: str, threshold: float = 0.85) -> list[LabelMatch]:
    """Every field `text` could be naming, best first.

    Returns all of them rather than the first: which one is right often
    depends on where the text sits and what value it points at, and that is
    the scoring stage's decision, not this one's.
    """
    matches = [
        match for match in (match_label(text, spec, threshold) for spec in ALL_FIELDS)
        if match is not None
    ]
    matches.sort(key=lambda match: (-match.score, FIELD_BY_NAME[match.field].priority))
    return matches


def best_match(text: str, threshold: float = 0.85) -> LabelMatch | None:
    matches = match_any(text, threshold)
    return matches[0] if matches else None


def is_negative_label(text: str) -> bool:
    """Whether this text must never be read as a money field."""
    candidate = normalise(text)
    return any(normalise(negative) in candidate for negative in MONEY_NEGATIVES)
