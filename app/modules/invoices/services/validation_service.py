"""Validation: does this extraction hold together as an invoice?

Extraction can be individually plausible and collectively impossible — every
field read confidently, and the taxes not summing to the total. That is the
most valuable signal the system has, because it catches errors no per-field
confidence can: a correctly-read subtotal paired with a misread total looks
perfect field by field and wrong the moment you add them up.

Findings are graded, and the grading carries real consequences:

  **error**   — arithmetic or format that cannot be right. Blocks automatic
                acceptance; a human must look.
  **warning** — suspicious but explicable. Lowers confidence, does not block.

The distinction matters because treating everything as an error sends every
invoice to review, which is the same as having no automation; and treating
everything as a warning lets a wrong total through silently.

**GSTIN checksums are a warning, not an error.** The algorithm is exact, so a
failure means either a genuinely invalid number or — far more often — OCR
misreading one character. Discarding the value would throw away fourteen
correct characters that a reviewer could fix in seconds.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from app.modules.invoices.dto import ParsedInvoiceResult

logger = logging.getLogger(__name__)

# Rounding tolerance for money comparisons. Invoices round per line and again
# at the total, so exact equality is the wrong test — a two-rupee gap across a
# twenty-line invoice is arithmetic, not an error.
MONEY_TOLERANCE = Decimal("2.00")
# Round-off is by definition sub-unit; more than this is a different field.
MAX_ROUND_OFF = Decimal("1.00")
# An invoice dated further ahead than this is a misread year.
FUTURE_TOLERANCE_DAYS = 2
# ...and one older than this is almost certainly a two-digit year misparse.
MAX_AGE_YEARS = 12

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

GSTIN_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_PATTERN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
PAN_PATTERN = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
IFSC_PATTERN = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

# GST state codes in use. 97 is "Other Territory", 99 "Centre Jurisdiction".
VALID_STATE_CODES: frozenset[str] = frozenset(
    [f"{code:02d}" for code in range(1, 39)] + ["97", "99"]
)

STATE_NAMES: dict[str, str] = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
    "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana", "07": "Delhi",
    "08": "Rajasthan", "09": "Uttar Pradesh", "10": "Bihar", "11": "Sikkim",
    "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur",
    "15": "Mizoram", "16": "Tripura", "17": "Meghalaya", "18": "Assam",
    "19": "West Bengal", "20": "Jharkhand", "21": "Odisha",
    "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "26": "Dadra and Nagar Haveli and Daman and Diu", "27": "Maharashtra",
    "29": "Karnataka", "30": "Goa", "31": "Lakshadweep", "32": "Kerala",
    "33": "Tamil Nadu", "34": "Puducherry", "35": "Andaman and Nicobar Islands",
    "36": "Telangana", "37": "Andhra Pradesh", "38": "Ladakh",
    "97": "Other Territory", "99": "Centre Jurisdiction",
}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    fields: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "fields": list(self.fields),
        }


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == SEVERITY_WARNING]

    @property
    def is_clean(self) -> bool:
        return not self.errors

    def fields_with_findings(self) -> set[str]:
        """Field names any finding implicates — the confidence stage lowers
        exactly these rather than penalising the whole document."""
        return {name for item in self.findings for name in item.fields}

    def add(self, code: str, severity: str, message: str, *fields: str) -> None:
        self.findings.append(Finding(code, severity, message, tuple(fields)))

    def to_dict(self) -> dict:
        return {
            "clean": self.is_clean,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "findings": [item.to_dict() for item in self.findings],
        }


# --- format checks ---------------------------------------------------------


def gstin_checksum_char(gstin: str) -> str | None:
    """The 15th character the first 14 imply, by the official mod-36 rule."""
    if len(gstin) < 14:
        return None
    total = 0
    for index, character in enumerate(gstin[:14].upper()):
        if character not in GSTIN_CHARSET:
            return None
        product = GSTIN_CHARSET.index(character) * (2 if index % 2 else 1)
        total += product // 36 + product % 36
    return GSTIN_CHARSET[(36 - total % 36) % 36]


def is_valid_gstin(gstin: str | None) -> bool:
    if not gstin:
        return False
    candidate = gstin.strip().upper()
    if not GSTIN_PATTERN.match(candidate):
        return False
    if candidate[:2] not in VALID_STATE_CODES:
        return False
    return gstin_checksum_char(candidate) == candidate[14]


def state_of(gstin: str | None) -> str | None:
    if not gstin or len(gstin) < 2:
        return None
    return STATE_NAMES.get(gstin[:2].strip())


# --- the checks ------------------------------------------------------------


def _check_required(parsed: ParsedInvoiceResult, report: ValidationReport) -> None:
    if not parsed.invoice_number:
        report.add("MISSING_INVOICE_NUMBER", SEVERITY_ERROR,
                   "No invoice number could be found.", "invoice_number")
    if not parsed.invoice_date:
        report.add("MISSING_INVOICE_DATE", SEVERITY_ERROR,
                   "No invoice date could be found.", "invoice_date")
    if parsed.total_amount is None:
        report.add("MISSING_TOTAL", SEVERITY_ERROR,
                   "No total amount could be found.", "total_amount")
    if not parsed.vendor.name and not parsed.vendor.gstin:
        report.add("MISSING_VENDOR", SEVERITY_ERROR,
                   "The supplier could not be identified.", "vendor_name", "vendor_gstin")


def _check_identifiers(parsed: ParsedInvoiceResult, report: ValidationReport) -> None:
    for party, gstin, name in (
        ("supplier", parsed.vendor.gstin, "vendor_gstin"),
        ("customer", parsed.customer.gstin, "customer_gstin"),
    ):
        if not gstin:
            continue
        candidate = gstin.strip().upper()
        if not GSTIN_PATTERN.match(candidate):
            report.add("GSTIN_MALFORMED", SEVERITY_WARNING,
                       f"The {party} GSTIN '{gstin}' is not in the expected format.", name)
            continue
        if candidate[:2] not in VALID_STATE_CODES:
            report.add("GSTIN_BAD_STATE_CODE", SEVERITY_WARNING,
                       f"The {party} GSTIN starts with '{candidate[:2]}', which is not a "
                       "valid state code.", name)
            continue
        if gstin_checksum_char(candidate) != candidate[14]:
            # A warning, not an error: OCR misreading one character is far
            # more likely than a genuinely fake GSTIN, and the other fourteen
            # are worth keeping for a reviewer to correct.
            report.add("GSTIN_CHECKSUM_FAILED", SEVERITY_WARNING,
                       f"The {party} GSTIN '{candidate}' fails its check digit — "
                       "one character may have been misread.", name)

    if (parsed.vendor.gstin and parsed.customer.gstin
            and parsed.vendor.gstin.strip().upper() == parsed.customer.gstin.strip().upper()):
        report.add("SAME_GSTIN_BOTH_PARTIES", SEVERITY_ERROR,
                   "The supplier and customer have the same GSTIN; one of them was "
                   "read from the wrong block.", "vendor_gstin", "customer_gstin")

    for pan, name in ((parsed.vendor.pan, "vendor_pan"), (parsed.customer.pan, "customer_pan")):
        if pan and not PAN_PATTERN.match(pan.strip().upper()):
            report.add("PAN_MALFORMED", SEVERITY_WARNING,
                       f"'{pan}' is not a valid PAN.", name)


def _check_dates(parsed: ParsedInvoiceResult, report: ValidationReport,
                 today: date | None = None) -> None:
    today = today or date.today()

    if parsed.invoice_date:
        if parsed.invoice_date > today + timedelta(days=FUTURE_TOLERANCE_DAYS):
            report.add("INVOICE_DATE_IN_FUTURE", SEVERITY_ERROR,
                       f"The invoice date {parsed.invoice_date.isoformat()} is in the future.",
                       "invoice_date")
        elif parsed.invoice_date < today - timedelta(days=365 * MAX_AGE_YEARS):
            report.add("INVOICE_DATE_IMPLAUSIBLY_OLD", SEVERITY_WARNING,
                       f"The invoice date {parsed.invoice_date.isoformat()} is more than "
                       f"{MAX_AGE_YEARS} years ago; the year may have been misread.",
                       "invoice_date")

    if parsed.invoice_date and parsed.due_date and parsed.due_date < parsed.invoice_date:
        report.add("DUE_BEFORE_INVOICE", SEVERITY_ERROR,
                   f"The due date {parsed.due_date.isoformat()} falls before the invoice "
                   f"date {parsed.invoice_date.isoformat()}.", "due_date", "invoice_date")

    if (parsed.purchase_order_date and parsed.invoice_date
            and parsed.purchase_order_date > parsed.invoice_date):
        report.add("PO_AFTER_INVOICE", SEVERITY_WARNING,
                   "The purchase order is dated after the invoice.",
                   "purchase_order_date", "invoice_date")


def _check_tax_structure(parsed: ParsedInvoiceResult, report: ValidationReport) -> None:
    cgst, sgst, igst = parsed.cgst_amount, parsed.sgst_amount, parsed.igst_amount

    # CGST and SGST are always levied together and always equal.
    if cgst is not None and sgst is not None and abs(cgst - sgst) > MONEY_TOLERANCE:
        report.add("CGST_SGST_MISMATCH", SEVERITY_ERROR,
                   f"CGST ({cgst}) and SGST ({sgst}) differ; on an intra-state invoice "
                   "they are always equal.", "cgst_amount", "sgst_amount")
    if cgst is not None and sgst is None:
        report.add("SGST_MISSING", SEVERITY_WARNING,
                   "CGST was found but SGST was not; they are always levied together.",
                   "sgst_amount")
    if sgst is not None and cgst is None:
        report.add("CGST_MISSING", SEVERITY_WARNING,
                   "SGST was found but CGST was not; they are always levied together.",
                   "cgst_amount")

    # IGST is inter-state; CGST/SGST intra-state. Both is a contradiction.
    if igst is not None and igst > 0 and ((cgst or 0) > 0 or (sgst or 0) > 0):
        report.add("IGST_WITH_CGST_SGST", SEVERITY_ERROR,
                   "Both IGST and CGST/SGST carry amounts; a supply is either "
                   "inter-state or intra-state, not both.",
                   "igst_amount", "cgst_amount", "sgst_amount")

    components = [value for value in (cgst, sgst, igst, getattr(parsed, "cess_amount", None))
                  if value is not None]
    if components and parsed.tax_amount is not None:
        if abs(sum(components) - parsed.tax_amount) > MONEY_TOLERANCE:
            report.add("TAX_COMPONENTS_MISMATCH", SEVERITY_ERROR,
                       f"The tax components total {sum(components)} but the tax amount "
                       f"reads {parsed.tax_amount}.", "tax_amount")


def _check_totals(parsed: ParsedInvoiceResult, report: ValidationReport) -> None:
    base = parsed.taxable_amount if parsed.taxable_amount is not None else parsed.subtotal

    if base is not None and parsed.total_amount is not None:
        expected = (
            base
            - (parsed.discount_amount or Decimal("0"))
            + (parsed.tax_amount or Decimal("0"))
            + (getattr(parsed, "freight_amount", None) or Decimal("0"))
            + (parsed.round_off or Decimal("0"))
        )
        difference = abs(expected - parsed.total_amount)
        if difference > MONEY_TOLERANCE:
            report.add("TOTAL_DOES_NOT_RECONCILE", SEVERITY_ERROR,
                       f"{base} less discount plus tax and round-off comes to {expected}, "
                       f"but the total reads {parsed.total_amount} "
                       f"(a difference of {difference}).",
                       "total_amount", "subtotal", "taxable_amount", "tax_amount")

    if parsed.round_off is not None and abs(parsed.round_off) > MAX_ROUND_OFF:
        report.add("ROUND_OFF_TOO_LARGE", SEVERITY_WARNING,
                   f"The round-off reads {parsed.round_off}; a rounding adjustment is "
                   "always under one unit, so this is probably a different field.",
                   "round_off")

    if parsed.total_amount is not None and parsed.total_amount < 0:
        report.add("NEGATIVE_TOTAL", SEVERITY_ERROR,
                   f"The total is negative ({parsed.total_amount}).", "total_amount")

    if (parsed.amount_paid is not None and parsed.amount_due is not None
            and parsed.total_amount is not None):
        if abs((parsed.amount_paid + parsed.amount_due) - parsed.total_amount) > MONEY_TOLERANCE:
            report.add("PAID_PLUS_DUE_MISMATCH", SEVERITY_WARNING,
                       "Amount paid plus amount due does not equal the total.",
                       "amount_paid", "amount_due")

    if (parsed.subtotal is not None and parsed.total_amount is not None
            and parsed.subtotal > parsed.total_amount + MONEY_TOLERANCE):
        report.add("SUBTOTAL_EXCEEDS_TOTAL", SEVERITY_ERROR,
                   f"The subtotal ({parsed.subtotal}) is greater than the total "
                   f"({parsed.total_amount}).", "subtotal", "total_amount")


def _check_line_items(parsed: ParsedInvoiceResult, report: ValidationReport) -> None:
    if not parsed.line_items:
        report.add("NO_LINE_ITEMS", SEVERITY_WARNING,
                   "No line items could be extracted.", "line_items")
        return

    for index, item in enumerate(parsed.line_items, start=1):
        label = item.description or f"line {index}"

        if item.quantity is not None and item.unit_price is not None:
            expected = item.quantity * item.unit_price
            reference = item.taxable_value if item.taxable_value is not None else item.total_amount
            if reference is not None and abs(expected - reference) > MONEY_TOLERANCE:
                report.add("LINE_ARITHMETIC_MISMATCH", SEVERITY_WARNING,
                           f"'{label}': quantity x rate is {expected}, but the line "
                           f"amount reads {reference}.", "line_items")

        if item.quantity is not None and item.quantity <= 0:
            report.add("LINE_NON_POSITIVE_QUANTITY", SEVERITY_WARNING,
                       f"'{label}' has a quantity of {item.quantity}.", "line_items")

        if item.hsn_sac and not re.fullmatch(r"\d{4}|\d{6}|\d{8}", item.hsn_sac.strip()):
            report.add("HSN_MALFORMED", SEVERITY_WARNING,
                       f"'{label}' has HSN/SAC '{item.hsn_sac}', which is not 4, 6 or 8 digits.",
                       "line_items")

    # The strongest structural check available: the lines must add up to the
    # invoice's own base. It catches a dropped row, which no per-line check can.
    line_values = [
        item.taxable_value if item.taxable_value is not None else item.total_amount
        for item in parsed.line_items
    ]
    if all(value is not None for value in line_values) and line_values:
        base = parsed.taxable_amount if parsed.taxable_amount is not None else parsed.subtotal
        if base is not None and abs(sum(line_values) - base) > MONEY_TOLERANCE:
            report.add("LINE_ITEMS_DO_NOT_SUM", SEVERITY_ERROR,
                       f"The line items total {sum(line_values)} but the invoice's "
                       f"taxable value reads {base}; a row may be missing or misread.",
                       "line_items", "subtotal", "taxable_amount")


def _check_place_of_supply(parsed: ParsedInvoiceResult, report: ValidationReport) -> None:
    """Cross-check the tax structure against where the parties are.

    Same state means CGST/SGST; different states mean IGST. This catches a
    misattributed party block, which is otherwise invisible: both GSTINs read
    perfectly, they were just assigned to the wrong sides.
    """
    supplier_state = state_of(parsed.vendor.gstin)
    customer_state = state_of(parsed.customer.gstin)
    if not supplier_state or not customer_state:
        return

    same_state = parsed.vendor.gstin[:2] == parsed.customer.gstin[:2]
    has_igst = (parsed.igst_amount or 0) > 0
    has_intra = (parsed.cgst_amount or 0) > 0 or (parsed.sgst_amount or 0) > 0

    if same_state and has_igst:
        report.add("IGST_ON_INTRA_STATE_SUPPLY", SEVERITY_WARNING,
                   f"Both parties are in {supplier_state}, but IGST was charged — "
                   "intra-state supplies carry CGST and SGST.",
                   "igst_amount", "vendor_gstin", "customer_gstin")
    if not same_state and has_intra:
        report.add("CGST_SGST_ON_INTER_STATE_SUPPLY", SEVERITY_WARNING,
                   f"The supplier is in {supplier_state} and the customer in "
                   f"{customer_state}, but CGST/SGST was charged — inter-state "
                   "supplies carry IGST.",
                   "cgst_amount", "sgst_amount", "vendor_gstin", "customer_gstin")


def validate(parsed: ParsedInvoiceResult, today: date | None = None) -> ValidationReport:
    """Run every check over a parsed invoice."""
    report = ValidationReport()

    _check_required(parsed, report)
    _check_identifiers(parsed, report)
    _check_dates(parsed, report, today)
    _check_tax_structure(parsed, report)
    _check_totals(parsed, report)
    _check_line_items(parsed, report)
    _check_place_of_supply(parsed, report)

    logger.info(
        "validation.completed errors=%d warnings=%d codes=%s",
        len(report.errors), len(report.warnings),
        [item.code for item in report.findings],
    )
    return report
