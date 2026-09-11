"""Validation: does this extraction hold together as a Bill of Entry?

The same principle as `invoices/services/validation_service.py` -- fields can
each be individually plausible and collectively impossible -- but the
arithmetic is customs arithmetic, and on this document class it is unusually
strong. A BoE is a tax computation printed alongside its own inputs, so most
of the numbers on it are checkable against the others:

    SWS        = 10% of BCD                       (statutory, exact)
    TOTAL DUTY = BCD + ACD + SWS + ... + G.CESS   (structural, exact)
    IGST       = a statutory GST rate applied to the grossed-up value
    amount     = unit price x quantity            (per goods line, exact)

**Why this matters more than OCR confidence.** On the sample document the
recogniser returned an assessable value of ``5525826`` against a true
``5528626`` -- a transposed digit -- at 0.898 confidence, comfortably above
any threshold that would not also reject most correct reads. Nothing in the
per-field confidence could distinguish it from a good read. The arithmetic
can: 10% of the misread value is 552582.6, and the BCD printed beside it is
552862.7, so the relationship fails and the field is flagged. Every check
here exists because a plausible-looking wrong number is the failure mode this
document class actually has.

Findings are graded exactly as the invoice validator grades them, and reuse
its `Finding` / `ValidationReport` types so a review screen renders both
document kinds through one code path.
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from decimal import Decimal

from app.modules.customs.dto import ParsedBillOfEntry
from app.modules.invoices.services.validation_service import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    ValidationReport,
    is_valid_gstin,
)

logger = logging.getLogger(__name__)

# Money tolerance. Customs rounds each head to the rupee and again at the
# total, so exact equality is the wrong test across a multi-head computation.
MONEY_TOLERANCE = Decimal("2.00")
# Per-line tolerance is tighter: unit price x quantity is computed to six
# decimals and printed to two, so the only legitimate gap is rounding.
LINE_TOLERANCE = Decimal("0.05")
# Social Welfare Surcharge is 10% of the aggregate customs duty, by statute.
SWS_RATE = Decimal("0.10")
SWS_TOLERANCE = Decimal("1.00")
# The GST rates IGST on an import can legally be charged at.
STATUTORY_GST_RATES = (
    Decimal("0"), Decimal("0.0025"), Decimal("0.03"), Decimal("0.05"),
    Decimal("0.12"), Decimal("0.18"), Decimal("0.28"),
)
GST_RATE_TOLERANCE = Decimal("0.004")
# How far the printed IGST may sit from the statutory rate applied to the
# taxable base. ICEGATE computes it exactly and rounds to the rupee, so the
# only legitimate gap is that rounding -- across the handful of heads that
# make up the base, a few rupees.
IGST_AMOUNT_TOLERANCE = Decimal("5.00")

# A BoE filed further ahead than this is a misread year.
FUTURE_TOLERANCE_DAYS = 2
MAX_AGE_YEARS = 12

CTH_PATTERN = re.compile(r"^\d{8}$")
PAN_PATTERN = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")


def _close(a: Decimal, b: Decimal, tolerance: Decimal) -> bool:
    return abs(a - b) <= tolerance


# --- completeness ----------------------------------------------------------


def _check_required(parsed: ParsedBillOfEntry, report: ValidationReport) -> None:
    if not parsed.be_number:
        report.add(
            "boe_number_missing", SEVERITY_ERROR,
            "Bill of Entry number could not be detected.", "be_number",
        )
    if not parsed.be_date:
        report.add(
            "boe_date_missing", SEVERITY_ERROR,
            "Bill of Entry date could not be detected.", "be_date",
        )
    if not parsed.invoices:
        report.add(
            "boe_no_invoices", SEVERITY_ERROR,
            "No Part-II invoice pages were found in this document.", "invoices",
        )
    if not parsed.iec and not parsed.gstin:
        report.add(
            "boe_importer_unidentified", SEVERITY_WARNING,
            "Importer could not be identified: neither IEC nor GSTIN was read.",
            "iec", "gstin",
        )


def _check_completeness(parsed: ParsedBillOfEntry, report: ValidationReport) -> None:
    """Did we read the whole document, or only the part the page cap allowed?

    This is the single most important check here, and it has no invoice
    equivalent. A Bill of Entry declares how many invoices it carries, and
    each Part-II page numbers itself "n of N". A run that stops early
    produces a result that is internally consistent and quietly missing most
    of the goods -- the default ``ocr_max_pages`` of 3 does exactly that to
    this seven-invoice document. Graded an error because the resulting
    duty-per-item figures would be wrong, not merely incomplete.
    """
    missing = parsed.missing_invoice_sequences
    if missing:
        expected = max(
            (inv.sequence_total for inv in parsed.invoices if inv.sequence_total),
            default=parsed.declared_invoice_count,
        )
        report.add(
            "boe_invoices_missing", SEVERITY_ERROR,
            f"Only {len(parsed.invoices)} of {expected} declared invoices were read; "
            f"missing invoice page(s) {', '.join(str(n) for n in missing)}. "
            "Raise ocr_max_pages or re-run over the full document.",
            "invoices",
        )

    declared = parsed.declared_invoice_count
    if declared is not None and not missing and len(parsed.invoices) != declared:
        report.add(
            "boe_invoice_count_mismatch", SEVERITY_WARNING,
            f"Header declares {declared} invoices but {len(parsed.invoices)} were parsed.",
            "declared_invoice_count",
        )

    if parsed.declared_item_count is not None and parsed.total_line_items:
        if parsed.total_line_items != parsed.declared_item_count:
            report.add(
                "boe_item_count_mismatch", SEVERITY_WARNING,
                f"Header declares {parsed.declared_item_count} items but "
                f"{parsed.total_line_items} goods lines were parsed.",
                "declared_item_count",
            )

    if parsed.duplicate_pages:
        report.add(
            "boe_duplicate_pages", SEVERITY_WARNING,
            f"{len(parsed.duplicate_pages)} page(s) repeated an already-read printed "
            "page and were skipped; check the scan for duplicated sheets.",
            "duplicate_pages",
        )


# --- duty arithmetic -------------------------------------------------------


def _check_duty(parsed: ParsedBillOfEntry, report: ValidationReport) -> None:
    duty = parsed.duty

    if duty.total_duty is not None and duty.components:
        component_sum = sum(duty.components)
        if not _close(component_sum, duty.total_duty, MONEY_TOLERANCE):
            report.add(
                "boe_duty_sum_mismatch", SEVERITY_ERROR,
                f"Duty heads add up to {component_sum} but total duty reads "
                f"{duty.total_duty}. One of them was misread.",
                "duty.total_duty",
            )

    # SWS is 10% of BCD by statute -- the tightest check on the page, and the
    # one that catches a transposed digit in either field.
    if duty.bcd is not None and duty.sws is not None:
        expected = duty.bcd * SWS_RATE
        if not _close(expected, duty.sws, SWS_TOLERANCE):
            report.add(
                "boe_sws_rate_mismatch", SEVERITY_ERROR,
                f"Social Welfare Surcharge should be 10% of BCD ({expected:.2f}) "
                f"but reads {duty.sws}.",
                "duty.sws", "duty.bcd",
            )

    _check_igst_rate(parsed, report)

    if duty.total_amount is not None and duty.total_duty is not None:
        extras = sum(
            value for value in (duty.interest, duty.penalty, duty.fine)
            if value is not None
        )
        expected = duty.total_duty + extras
        if not _close(expected, duty.total_amount, MONEY_TOLERANCE):
            report.add(
                "boe_total_amount_mismatch", SEVERITY_WARNING,
                f"Total amount ({duty.total_amount}) does not match total duty plus "
                f"interest, penalty and fine ({expected}).",
                "duty.total_amount",
            )


def _check_igst_rate(parsed: ParsedBillOfEntry, report: ValidationReport) -> None:
    """IGST must work out to a rate the law actually allows.

    Import IGST is charged on the assessable value grossed up by the customs
    duties. Recovering the implied rate and checking it against the statutory
    list catches a misread in any of the three inputs, without needing to
    know which rate this particular tariff head attracts.
    """
    duty = parsed.duty
    if duty.igst is None or duty.assessable_value is None:
        return
    base = duty.assessable_value + sum(
        value for value in (duty.bcd, duty.sws, duty.acd, duty.cvd, duty.nccd)
        if value is not None
    )
    if base <= 0:
        return

    # Compared as an amount, not as a rate. The rate form is far too blunt to
    # be worth running: on the sample, a 2,800-rupee error in a 5.5-crore
    # assessable value moves the implied rate from 18.000% to 18.008%, which
    # no tolerance loose enough for real rounding would ever reject. The same
    # error moves the *expected IGST* by 504 rupees, against an ICEGATE
    # computation that is exact to the rupee. So the check asks the question
    # that has a sharp answer: given this base, is the printed IGST what the
    # nearest statutory rate would actually produce?
    best_rate, best_expected, best_gap = None, None, None
    for rate in STATUTORY_GST_RATES:
        expected = base * rate
        gap = abs(expected - duty.igst)
        if best_gap is None or gap < best_gap:
            best_rate, best_expected, best_gap = rate, expected, gap

    if best_gap is not None and best_gap <= IGST_AMOUNT_TOLERANCE:
        return

    report.add(
        "boe_igst_amount_mismatch", SEVERITY_ERROR,
        f"IGST reads {duty.igst}, but the nearest statutory rate "
        f"({best_rate * 100:.2f}%) on a taxable base of {base} gives "
        f"{best_expected:.2f} -- a gap of {best_gap:.2f}. One of the assessable "
        "value, the duty heads or the IGST was misread.",
        "duty.igst", "duty.assessable_value",
    )


def _check_assessable_values(parsed: ParsedBillOfEntry, report: ValidationReport) -> None:
    """Per-invoice assessable values should add up to the document total."""
    per_invoice = [
        invoice.assessable_value for invoice in parsed.invoices
        if invoice.assessable_value is not None
    ]
    total = parsed.duty.assessable_value
    if total is None or not per_invoice:
        return
    if len(per_invoice) != len(parsed.invoices):
        return  # a page failed to yield its value; not a document error
    if parsed.missing_invoice_sequences:
        # Invoices are missing, so of course the parts do not sum to the
        # whole. `_check_completeness` has already said so; repeating it
        # here as an arithmetic failure sends a reviewer looking for a
        # misread number that does not exist.
        return
    combined = sum(per_invoice)
    if not _close(combined, total, MONEY_TOLERANCE):
        report.add(
            "boe_assessable_value_mismatch", SEVERITY_ERROR,
            f"Invoice assessable values add up to {combined} but the declared "
            f"total assessable value is {total}.",
            "duty.assessable_value",
        )


# --- goods lines -----------------------------------------------------------


def _check_line_items(parsed: ParsedBillOfEntry, report: ValidationReport) -> None:
    """Per-line arithmetic, which also proves the column zip was correct.

    `parser._parse_line_items` recovers each column as its own token run and
    zips them by index. If that zip slipped -- a dropped quantity shifting
    every later line by one -- the values stay individually plausible and
    only the multiplication reveals it. So this check is doing double duty:
    validating the document, and validating the parse.
    """
    for invoice in parsed.invoices:
        label = invoice.invoice_number or invoice.sequence or "unknown"
        for item in invoice.line_items:
            if item.cth and not CTH_PATTERN.match(item.cth):
                report.add(
                    "boe_cth_malformed", SEVERITY_WARNING,
                    f"Invoice {label}: tariff head '{item.cth}' is not eight digits.",
                    "line_items.cth",
                )
            if item.unit_price is None or item.quantity is None or item.amount is None:
                report.add(
                    "boe_line_incomplete", SEVERITY_WARNING,
                    f"Invoice {label}: goods line "
                    f"'{(item.description or item.cth or '?')[:40]}' is missing a "
                    "price, quantity or amount.",
                    "line_items",
                )
                continue
            expected = item.unit_price * item.quantity
            if not _close(expected, item.amount, LINE_TOLERANCE):
                report.add(
                    "boe_line_arithmetic", SEVERITY_ERROR,
                    f"Invoice {label}: {item.unit_price} x {item.quantity} = "
                    f"{expected} but the line amount reads {item.amount}.",
                    "line_items",
                )

        if invoice.line_items and invoice.assessable_value is not None:
            amounts = [
                item.amount for item in invoice.line_items if item.amount is not None
            ]
            if len(amounts) == len(invoice.line_items) and invoice.invoice_value is not None:
                combined = sum(amounts)
                if not _close(combined, invoice.invoice_value, MONEY_TOLERANCE):
                    report.add(
                        "boe_invoice_value_mismatch", SEVERITY_WARNING,
                        f"Invoice {label}: goods lines add up to {combined} but the "
                        f"declared invoice value is {invoice.invoice_value}.",
                        "invoices.invoice_value",
                    )


# --- identifiers and dates -------------------------------------------------


def _check_identifiers(parsed: ParsedBillOfEntry, report: ValidationReport) -> None:
    if parsed.gstin and not is_valid_gstin(parsed.gstin):
        report.add(
            "boe_gstin_checksum", SEVERITY_WARNING,
            f"GSTIN '{parsed.gstin}' fails its checksum; most likely one "
            "character was misread.",
            "gstin",
        )
    if parsed.iec and not PAN_PATTERN.match(parsed.iec):
        report.add(
            "boe_iec_malformed", SEVERITY_WARNING,
            f"IEC '{parsed.iec}' is not in PAN format.", "iec",
        )
    if parsed.be_number and not re.fullmatch(r"\d{7}", parsed.be_number):
        report.add(
            "boe_number_malformed", SEVERITY_WARNING,
            f"Bill of Entry number '{parsed.be_number}' is not seven digits.",
            "be_number",
        )
    # The importer's GSTIN embeds their PAN, which is also their IEC.
    if parsed.gstin and parsed.iec and parsed.gstin[2:12] != parsed.iec:
        report.add(
            "boe_iec_gstin_disagree", SEVERITY_WARNING,
            f"IEC '{parsed.iec}' does not match the PAN embedded in GSTIN "
            f"'{parsed.gstin}'; one of the two was misread.",
            "iec", "gstin",
        )


def _check_dates(
    parsed: ParsedBillOfEntry, report: ValidationReport, today: date | None
) -> None:
    reference = today or date.today()
    if parsed.be_date:
        if parsed.be_date > reference + timedelta(days=FUTURE_TOLERANCE_DAYS):
            report.add(
                "boe_date_future", SEVERITY_WARNING,
                f"Bill of Entry date {parsed.be_date.isoformat()} is in the future.",
                "be_date",
            )
        elif parsed.be_date < reference - timedelta(days=365 * MAX_AGE_YEARS):
            report.add(
                "boe_date_implausible", SEVERITY_WARNING,
                f"Bill of Entry date {parsed.be_date.isoformat()} is implausibly old; "
                "the year was probably misread.",
                "be_date",
            )
    for invoice in parsed.invoices:
        if invoice.invoice_date and parsed.be_date and invoice.invoice_date > parsed.be_date:
            report.add(
                "boe_invoice_after_filing", SEVERITY_WARNING,
                f"Invoice {invoice.invoice_number or invoice.sequence} is dated "
                f"{invoice.invoice_date.isoformat()}, after the Bill of Entry date "
                f"{parsed.be_date.isoformat()}.",
                "invoices.invoice_date",
            )


# --- entry point -----------------------------------------------------------


def validate(parsed: ParsedBillOfEntry, today: date | None = None) -> ValidationReport:
    """Run every check over a parsed Bill of Entry."""
    report = ValidationReport()

    _check_required(parsed, report)
    _check_completeness(parsed, report)
    _check_duty(parsed, report)
    _check_assessable_values(parsed, report)
    _check_line_items(parsed, report)
    _check_identifiers(parsed, report)
    _check_dates(parsed, report, today)

    logger.info(
        "boe_validation.completed errors=%d warnings=%d codes=%s",
        len(report.errors), len(report.warnings),
        [item.code for item in report.findings],
    )
    return report
