"""The Bill of Entry parser.

Mirrors `BaseInvoiceParser`'s contract -- `can_parse(text)` then
`parse(ExtractedDocument)` -- so it can be driven by the same orchestration,
but it returns a `ParsedBillOfEntry` rather than a `ParsedInvoiceResult`.
See `patterns.py` for why extraction is shape-driven rather than
label-driven, and `dto.py` for why the result type is separate.

**What this parser will and will not claim.**

Part-II pages (the commercial content: which invoice, whose goods, how many,
at what price) survive OCR in close to reading order, and are extracted
field by field with good confidence.

Part-I is a dense ruled grid whose cells are frequently blank. A blank cell
emits no token at all, so counting values off against labels positionally --
the obvious approach -- silently shifts every field after the first gap. This
parser therefore recovers Part-I fields only where the *value itself* is
identifiable by shape (a GSTIN, a port code, an eight-digit CTH), and leaves
the rest to be confirmed arithmetically in `validation.py`. Where it cannot
tell, it records nothing rather than guessing: a missing field is visible to
a reviewer, a confidently wrong one is not.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from . import patterns as pat
from .dto import (
    BoeDuty,
    BoeInvoiceRef,
    BoeLineItem,
    BoeManifest,
    ParsedBillOfEntry,
)

logger = logging.getLogger(__name__)

# A description fragment that opens a new goods line. ICEGATE part numbers
# lead with a short alpha group, a number, then a comma or dash --
# "WG-4187A,XSYTIN-1,INSERT", "RCGN-4VA,...". Used only to split an already
# item-scoped run of fragments, never to decide what is a description.
_PART_NUMBER_HEAD = re.compile(r"^[A-Z]{2,5}[- ]?\d+[A-Z]*\s*[,\-]")

# Where the goods table starts on a Part-II page. Any of these will do; OCR
# loses different ones on different pages, and on the sample's last page the
# whole "1.S NO. 2.CTH 3.DESCRIPTION" row came back as "T DO UI 9 AN". The
# MISC CHARGE / ASS. VALUE row is the last header line before the goods and
# survives when the table header does not, so it is included as a backstop --
# the assessable value it introduces is excluded from the goods amounts by
# `_known_amounts`.
_ITEM_TABLE_HEADERS = (
    re.compile(r"2\s*\.?\s*CTH", re.IGNORECASE),
    re.compile(r"3\s*\.?\s*DESCRIPTION", re.IGNORECASE),
    re.compile(r"4\s*\.?\s*UNIT\s*PRICE", re.IGNORECASE),
    re.compile(r"13\s*\.?\s*MISC\s*CHARGE", re.IGNORECASE),
)

# A duty head below this is a flag or a column number, not money. The duty
# row sits among single-digit status columns, and without a floor the
# statutory 10% test finds (2, 1) on every document.
MINIMUM_DUTY_HEAD = Decimal("100")
# How far SWS may sit from exactly 10% of BCD, proportionally. Generous
# enough for the rupee rounding customs applies, tight enough that a
# transposed digit fails.
SWS_MATCH_RATIO = Decimal("0.001")
# Absolute tolerance when closing components against a printed total.
TOTAL_TOLERANCE = Decimal("2.00")
# The assessable value is a multiple of the duty charged on it, never orders
# of magnitude more. Bounds the search so a bond or challan number from the
# next section of the form cannot be picked up as the goods value.
ASSESSABLE_VALUE_MAX_MULTIPLE = Decimal("100")

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# --- scalar coercion -------------------------------------------------------


def to_decimal(raw: str | None) -> Decimal | None:
    """Parse an ICEGATE numeric token. Never raises; None means "not a number"."""
    if not raw:
        return None
    try:
        return Decimal(raw.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None


def to_date(raw: str | None) -> date | None:
    """Parse the two date formats an ICEGATE print uses.

    Two-digit years are read as 20xx. A Bill of Entry is a customs filing for
    goods physically in transit, so a 19xx reading is never the right one.
    """
    if not raw:
        return None
    text = raw.strip()
    match = pat.DATE_DMY_ALPHA.fullmatch(text)
    if match:
        day, month, year = text.split("-")
        month_number = _MONTHS.get(month.upper()[:3])
        if not month_number:
            return None
        year_number = int(year)
        if year_number < 100:
            year_number += 2000
        try:
            return date(year_number, month_number, int(day))
        except ValueError:
            return None
    if pat.DATE_SLASH.fullmatch(text):
        for fmt in ("%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    return None


# --- page handling ---------------------------------------------------------


def split_pages(text: str) -> list[str]:
    """Split a flat extraction back into pages on the ICEGATE page footer.

    `ExtractedDocument.text` joins pages with a newline and no marker, so the
    page a value came from would otherwise be unrecoverable -- and on this
    document the page *is* the record boundary: one Part-II page is one
    commercial invoice. The footer ("Page 3 Of 19") is printed on every page
    and survives OCR reliably, so it is used as the separator.

    Pages are kept with their footer so the caller can read the page number
    back off them. Text before the first footer is page one.
    """
    if not text:
        return []
    boundaries = [match.end() for match in pat.PAGE_FOOTER.finditer(text)]
    if not boundaries:
        return [text]
    pages, start = [], 0
    for end in boundaries:
        pages.append(text[start:end])
        start = end
    remainder = text[start:]
    if remainder.strip():
        # Trailing text after the last footer is the tail of that same page,
        # not a new one -- OCR often emits the "Verify using ICETRAK" strap
        # after the page number. Appending rather than adding a page keeps
        # the parsed-page count honest.
        if pages:
            pages[-1] += remainder
        else:
            pages.append(remainder)
    return pages


def page_number(page_text: str) -> int | None:
    match = pat.PAGE_FOOTER.search(page_text)
    return int(match.group(1)) if match else None


def _non_label_lines(page_text: str) -> list[str]:
    return [line.strip() for line in page_text.splitlines() if not pat.is_label(line)]


# --- Part II: one commercial invoice per page ------------------------------


def _item_table_offset(page_text: str) -> int:
    """Character offset where the goods table begins, or -1.

    Everything before it belongs to the invoice header -- notably 14.ASS.
    VALUE, which is a two-decimal amount and would otherwise be swept up as
    a line amount.
    """
    offsets = [
        match.start()
        for header in _ITEM_TABLE_HEADERS
        for match in [header.search(page_text)]
        if match
    ]
    return min(offsets) if offsets else -1


def _parse_line_items(region: str, exclude: set[str] | None = None) -> list[BoeLineItem]:
    """Recover goods lines from the item region by zipping token streams.

    Each column arrives as its own run of tokens, in item order, interleaved
    with the other columns and with labels. Within a column the order is
    reliable even though the interleaving is not -- so the columns are
    collected separately and zipped by index.

    The zip is a hypothesis, not a result. `validation.py` checks
    ``unit_price * quantity == amount`` on every line, and a misalignment
    breaks that arithmetic loudly instead of producing a plausible-looking
    wrong quantity.
    """
    cths = pat.CTH.findall(region)
    if not cths:
        return []
    # Start at the first tariff head. When the goods-table header was lost to
    # OCR the caller passes the whole page, and everything above the first
    # CTH is invoice header that would otherwise pollute every token stream.
    region = region[region.index(cths[0]):]
    excluded = exclude or set()

    quantities: list[tuple[Decimal | None, str | None]] = []
    consumed: set[str] = set()
    for value, uqc in pat.QUANTITY_WITH_UQC.findall(region):
        if uqc.upper() in pat.UQC_CODES:
            quantities.append((to_decimal(value), uqc.upper()))
            consumed.add(value)

    unit_prices = [
        to_decimal(value) for value in pat.SIX_DECIMAL.findall(region)
        if value not in consumed
    ]
    amounts = [
        to_decimal(value) for value in pat.TWO_DECIMAL.findall(region)
        if value not in excluded and value.replace(",", "") not in excluded
    ]
    descriptions = _split_descriptions(region, len(cths))

    items: list[BoeLineItem] = []
    for index, cth in enumerate(cths):
        quantity, uqc = quantities[index] if index < len(quantities) else (None, None)
        items.append(
            BoeLineItem(
                serial=index + 1,
                cth=cth,
                description=descriptions[index] if index < len(descriptions) else None,
                unit_price=unit_prices[index] if index < len(unit_prices) else None,
                quantity=quantity,
                uqc=uqc,
                amount=amounts[index] if index < len(amounts) else None,
            )
        )
    return items


def _split_descriptions(region: str, expected: int) -> list[str]:
    """Group description fragments into one description per goods line.

    A description is printed over several lines and OCR scatters them among
    the numeric columns, so they are collected in order and then split where
    a fragment looks like the head of a new part number. When that yields the
    wrong number of groups the fragments are not force-fitted -- a wrong
    description attached to the right CTH is worse than none.
    """
    fragments = [
        line for line in _non_label_lines(region)
        if pat.DESCRIPTION_FRAGMENT.match(line)
        and not line.replace(".", "").replace(",", "").isdigit()
        and line.upper() not in pat.UQC_CODES
        and not pat.QUANTITY_WITH_UQC.fullmatch(line)
    ]
    if not fragments:
        return []

    groups: list[list[str]] = []
    for fragment in fragments:
        if _PART_NUMBER_HEAD.match(fragment) or not groups:
            groups.append([fragment])
        else:
            groups[-1].append(fragment)

    if expected and len(groups) != expected:
        logger.info(
            "boe.description_grouping_mismatch expected=%d got=%d", expected, len(groups)
        )
        if len(groups) < expected:
            return [" ".join(group) for group in groups]
    return [" ".join(group) for group in groups]


def parse_part_two_page(page_text: str, source_page: int | None = None) -> BoeInvoiceRef | None:
    """Parse one Part-II page into the invoice it declares."""
    banner = pat.PART_TWO_BANNER.search(page_text)
    if not banner:
        return None

    invoice = BoeInvoiceRef(
        sequence_index=int(banner.group(1)),
        sequence_total=int(banner.group(2)),
        source_page=source_page,
    )

    split_at = _item_table_offset(page_text)
    header_region = page_text[:split_at] if split_at > 0 else page_text
    item_region = page_text[split_at:] if split_at > 0 else ""

    # 14.ASS. VALUE is read by its own label rather than by position. OCR
    # emits it *after* the 3.DESCRIPTION header on this layout, so splitting
    # the page at the goods table leaves it on the wrong side -- and it is a
    # two-decimal amount, so it would then be consumed as the first line
    # amount and shift every goods line along by one.
    invoice.assessable_value = _labelled_amount(
        page_text, r"14\s*\.?\s*ASS\s*\.?\s*VALUE"
    )

    # Invoice serial, number and date follow the 2.INVOICE NO label as a
    # short token run. Anchoring on the label rather than scanning the whole
    # header matters: the repeated page-header strip carries the BE number,
    # which is also a bare digit run and would otherwise win.
    _read_invoice_identity(page_text, invoice)

    # Supplier block: the free-text run after the SUPPLIER NAME label.
    invoice.supplier_name, invoice.supplier_address = _supplier_from(header_region)
    for country in pat.COUNTRY_HINTS:
        if country in header_region.upper():
            invoice.supplier_country = country
            break

    for token in re.findall(r"\b([A-Z]{3})\b", header_region):
        if invoice.currency is None and token in pat.CURRENCIES:
            invoice.currency = token
        if invoice.incoterm is None and token in pat.INCOTERMS:
            invoice.incoterm = token

    invoice.invoice_value = _labelled_amount(
        header_region, r"1\s*\.?\s*INV\s*VALUE"
    )

    # Fall back to the whole page when the goods-table header did not
    # survive OCR -- on the sample's last page it came back as "T DO UI 9
    # AN" and an offset-based split would have found no goods at all.
    region = item_region or page_text
    invoice.line_items = _parse_line_items(region, exclude=_known_amounts(invoice))
    return invoice


def _known_amounts(invoice: BoeInvoiceRef) -> set[str]:
    """Amounts already claimed by a header field, so the goods-line zip
    does not consume them a second time."""
    claimed = set()
    for value in (invoice.assessable_value, invoice.invoice_value):
        if value is not None:
            claimed.add(f"{value:.2f}")
            claimed.add(str(value))
    return claimed


def _read_invoice_identity(page_text: str, invoice: BoeInvoiceRef) -> None:
    """Serial number, invoice number and invoice date, from the 2.INVOICE NO
    cell. Values follow the label row as a short run of standalone tokens."""
    match = re.search(r"2\s*\.?\s*INVO?I?CE\s*NO", page_text, re.IGNORECASE)
    window = page_text[match.end():match.end() + 500] if match else page_text
    for line in _non_label_lines(window):
        if invoice.invoice_date is None:
            found = pat.DATE_DMY_ALPHA.search(line)
            if found:
                invoice.invoice_date = to_date(found.group(1))
                continue
        if invoice.serial_no is None and re.fullmatch(r"\d{1,2}", line):
            invoice.serial_no = int(line)
            continue
        if invoice.invoice_number is None and re.fullmatch(r"\d{6,12}", line):
            invoice.invoice_number = line
        if invoice.invoice_number and invoice.invoice_date:
            return


def _supplier_from(header_region: str) -> tuple[str | None, str | None]:
    """Name and address from the 3.SUPPLIER NAME cell.

    The cell's own label is reliably recognised; what follows it, up to the
    next numbered label, is the value. This is the one place label adjacency
    does work on this document, because the cell holds a text block rather
    than a grid row.
    """
    match = re.search(r"3\s*\.?\s*SUPPLIER\s*NAME", header_region, re.IGNORECASE)
    if not match:
        return None, None
    # Consume the rest of the label's own line. OCR routinely runs the whole
    # header together -- "3.SUPPLIERNAME&ADDRESS/CLIENTDETAILS" -- and
    # resuming at the match end would take the tail of the label as the
    # supplier's name.
    line_end = header_region.find("\n", match.end())
    tail = header_region[line_end + 1:] if line_end != -1 else ""
    collected: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if pat.NUMBERED_LABEL.match(stripped):
            if collected:
                break
            continue
        if pat.PAGE_FOOTER.search(stripped):
            break
        collected.append(stripped)
        if len(collected) >= 5:
            break
    if not collected:
        return None, None
    return collected[0], " ".join(collected[1:]) or None


# --- Part I: the summary page ---------------------------------------------


def parse_part_one_page(page_text: str, result: ParsedBillOfEntry) -> None:
    """Fill the Part-I header fields that are identifiable by shape.

    Deliberately partial. Fields whose only distinguishing feature is grid
    position -- the twelve one-letter status flags, the blank duty heads --
    are not guessed at here; see the module docstring.
    """
    upper = page_text.upper()

    port = pat.PORT_CODE.search(upper)
    if port:
        result.port_code = port.group(0)

    iec = pat.IEC_WITH_BRANCH.search(upper)
    if iec:
        result.iec, result.iec_branch = iec.group(1), iec.group(2)
    else:
        # The IEC/Br cell is in the header strip, which is the most degraded
        # part of the page. The same number is repeated in the SEZ unit
        # details line ("IEC ABBCS5682H SAFRAN AIRCRAFT ENGINES"), which sits
        # in plain text and survives; take it from there when the cell fails.
        fallback = re.search(r"IEC\s*[:\-]?\s*([A-Z]{5}\d{4}[A-Z])\b", upper)
        if fallback:
            result.iec = fallback.group(1)

    gstin = pat.GSTIN.search(upper)
    if gstin:
        result.gstin = gstin.group(1)

    for candidate in pat.CB_CODE.findall(upper):
        if candidate != result.iec:
            result.cb_code = candidate
            break

    # BE number: a seven-digit run that is not part of a longer identifier
    # and not one of the amounts. Taken from the header strip only, which is
    # everything before the Part-I banner.
    banner = pat.PART_ONE_BANNER.search(page_text)
    header_strip = page_text[:banner.start()] if banner else page_text
    be_candidates = pat.BE_NUMBER.findall(header_strip)
    if be_candidates:
        result.be_number = be_candidates[0]

    dates = pat.DATE_SLASH.findall(header_strip)
    if dates:
        result.be_date = to_date(dates[0])

    rate = pat.EXCHANGE_RATE.search(upper)
    if rate:
        result.exchange_currency = rate.group(1).upper()
        result.exchange_rate = to_decimal(rate.group(2))

    for country in pat.COUNTRY_HINTS:
        if country in upper and country != "INDIA":
            result.country_of_origin = country
            break

    result.duty = _parse_duty(page_text, result)
    result.manifest = _parse_manifest(page_text)


def _parse_duty(page_text: str, result: ParsedBillOfEntry) -> BoeDuty:
    """Recover the duty block, and say so when it cannot be trusted.

    The duty row is the part of Part-I most damaged by the grid problem:
    blank heads emit nothing, so the Nth value is not the Nth label. And the
    two labels that would anchor it -- 14.TOTAL DUTY, 18.TOT ASS VAL -- are
    printed in the smallest type on the page and do not survive OCR; on the
    sample the former came back as "2ATAA".

    So the heads are identified by the relationships they must satisfy
    rather than by position or label:

        SWS   = 10% of BCD                      (statutory)
        TOTAL = BCD + SWS + IGST                (structural)

    Both at once, not one then the other. The statutory ratio alone is
    ambiguous on a real document: this consignment's tariff charges BCD at
    10% of the assessable value, so (ASS VAL, BCD) stands in exactly the
    same 10:1 ratio as (BCD, SWS) and is the larger, more tempting pair.
    Only the total-duty sum tells them apart. A combination that does not
    close is not reported at all -- `validation.py` would have no way to
    catch a self-consistent wrong answer.
    """
    duty = BoeDuty()

    # The component row is anchored on 1.BCD, which survives OCR far more
    # reliably than the labels beside it.
    anchor = re.search(r"1\s*\.?\s*BCD", page_text, re.IGNORECASE)
    if not anchor:
        return duty
    window = page_text[anchor.end():anchor.end() + 900]
    candidates = [to_decimal(value) for value in pat.LOOSE_AMOUNT.findall(window)]
    candidates = [value for value in candidates if value is not None and value > 0][:30]
    lookup = set(candidates)

    solution = _solve_duty_heads(candidates, lookup)
    if solution is None:
        result.add_warning(
            "Duty heads could not be reconciled: no combination of the amounts "
            "read satisfies the statutory SWS ratio and sums to a printed total. "
            "The duty breakdown needs manual entry."
        )
        return duty

    duty.bcd, duty.sws, duty.igst, duty.total_duty = solution

    # TOT ASS VAL is the last column of the same row, six columns past its
    # own label. It is taken as the largest amount left once the duty heads
    # are spoken for, bounded so that a bond or challan number strayed in
    # from the next section cannot be mistaken for it.
    claimed = {duty.bcd, duty.sws, duty.igst, duty.total_duty}
    ceiling = duty.total_duty * ASSESSABLE_VALUE_MAX_MULTIPLE
    remaining = [
        value for value in candidates
        if value not in claimed and duty.total_duty < value < ceiling
    ]
    duty.assessable_value = max(remaining) if remaining else None
    return duty


def _solve_duty_heads(
    candidates: list[Decimal], lookup: set[Decimal]
) -> tuple[Decimal, Decimal, Decimal, Decimal] | None:
    """Find (BCD, SWS, IGST, TOTAL) satisfying both duty relationships.

    Largest BCD first, because the heads are large and the row is littered
    with the single digits of the neighbouring flag columns. Returns None
    rather than a partial answer: half a duty block is not useful, and a
    guessed one is worse than none.
    """
    for bcd in sorted(candidates, reverse=True):
        if bcd < MINIMUM_DUTY_HEAD:
            break
        target = bcd / 10
        sws = next(
            (
                value for value in candidates
                if value < bcd and abs(value - target) <= target * SWS_MATCH_RATIO
            ),
            None,
        )
        if sws is None:
            continue
        base = bcd + sws
        for igst in candidates:
            if igst <= 0 or igst == bcd or igst == sws:
                continue
            total = base + igst
            match = next(
                (value for value in lookup if abs(value - total) <= TOTAL_TOLERANCE),
                None,
            )
            if match is not None and match > base:
                return bcd, sws, igst, match
    return None


def _close(a: Decimal | None, b: Decimal | None, tolerance: Decimal = Decimal("1.00")) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance


def _labelled_amount(page_text: str, label_pattern: str) -> Decimal | None:
    """The first amount appearing after `label_pattern`, within a short window.

    Bounded to a few hundred characters because on a grid page an unbounded
    search finds a number belonging to an entirely different section.
    """
    match = re.search(label_pattern, page_text, re.IGNORECASE)
    if not match:
        return None
    window = page_text[match.end():match.end() + 400]
    for line in window.splitlines():
        stripped = line.strip()
        if pat.is_label(stripped):
            continue
        found = pat.LOOSE_AMOUNT.search(stripped)
        if found:
            return to_decimal(found.group(1))
    return None


def _parse_manifest(page_text: str) -> BoeManifest:
    """The D.MANIFEST DETAILS row.

    Its values do arrive in column order -- the row is fully populated, so
    the blank-cell problem that defeats the duty block does not apply. The
    airway bill numbers are the long digit runs; the dates are dates.
    """
    manifest = BoeManifest()
    match = re.search(r"1\s*\.?\s*IGM\s*NO", page_text, re.IGNORECASE)
    if not match:
        return manifest
    window = page_text[match.end():match.end() + 900]

    dates = [to_date(token) for token in pat.ANY_DATE.findall(window)]
    dates = [value for value in dates if value is not None]
    if len(dates) >= 1:
        manifest.igm_date = dates[0]
    if len(dates) >= 2:
        manifest.inward_date = dates[1]
    if len(dates) >= 3:
        manifest.gigm_date = dates[2]
    if len(dates) >= 4:
        manifest.mawb_date = dates[3]
    if len(dates) >= 5:
        manifest.hawb_date = dates[4]

    # IGM numbers are seven digits; air waybill numbers are ten or eleven.
    long_runs = re.findall(r"(?<!\d)(\d{10,11})(?!\d)", window)
    if len(long_runs) >= 1:
        manifest.mawb_number = long_runs[0]
    if len(long_runs) >= 2:
        manifest.hawb_number = long_runs[1]

    igm = re.findall(r"(?<!\d)(\d{7})(?!\d)", window)
    if igm:
        manifest.igm_number = igm[0]
        if len(igm) > 1:
            manifest.gigm_number = igm[1]
    return manifest


# --- the parser ------------------------------------------------------------


class BillOfEntryParser:
    """Parses Indian customs Bills of Entry (ICEGATE print format)."""

    name = "bill_of_entry"
    version = "1.0"

    def can_parse(self, text: str) -> bool:
        """Claim only documents that actually look like a Bill of Entry.

        Two independent signals are required, not one: "INDIAN CUSTOMS"
        appears on several unrelated customs forms, and a lone "Page n Of m"
        footer means nothing. Requiring a document marker *and* a part banner
        keeps this parser from taking documents the invoice parsers handle
        better.
        """
        if not text or not text.strip():
            return False
        has_marker = any(marker.search(text) for marker in pat.BOE_MARKERS)
        has_part = bool(
            pat.PART_ONE_BANNER.search(text) or pat.PART_TWO_BANNER.search(text)
        )
        return has_marker and has_part

    def parse(self, extracted) -> ParsedBillOfEntry:
        """Parse an already-extracted document. No filesystem, no database."""
        result = ParsedBillOfEntry(
            parser_name=self.name,
            parser_version=self.version,
            used_ocr=getattr(extracted, "used_ocr", False),
        )
        pages = split_pages(getattr(extracted, "text", "") or "")
        result.page_count = getattr(extracted, "page_count", 0) or len(pages)

        seen_page_numbers: dict[int, int] = {}
        for index, page_text in enumerate(pages, start=1):
            printed = page_number(page_text)
            # A scanned BoE routinely contains the same sheet twice -- the
            # sample has page 6 scanned as both PDF page 6 and page 8. Left
            # unhandled that double-counts an invoice and its goods.
            if printed is not None and printed in seen_page_numbers:
                result.duplicate_pages.append(index)
                result.add_warning(
                    f"Page {index} repeats printed page {printed} "
                    f"(already read as page {seen_page_numbers[printed]}); it was skipped."
                )
                continue
            if printed is not None:
                seen_page_numbers[printed] = index

            result.pages_parsed += 1
            invoice = parse_part_two_page(page_text, source_page=index)
            if invoice is not None:
                result.invoices.append(invoice)
            elif pat.PART_ONE_BANNER.search(page_text):
                parse_part_one_page(page_text, result)

        self._cross_fill(result)
        result.parsing_confidence = self._score_confidence(result)
        logger.info(
            "boe.parsed be_number=%s invoices=%d line_items=%d pages=%d confidence=%.2f",
            result.be_number, len(result.invoices), result.total_line_items,
            result.pages_parsed, result.parsing_confidence,
        )
        return result

    @staticmethod
    def _cross_fill(result: ParsedBillOfEntry) -> None:
        """Fill header facts that only the Part-II pages carried.

        The declared invoice count is on Part-I, but the "n/7" markers say
        the same thing and survive OCR better. When Part-I was unreadable --
        or was never reached, because the page cap cut the document short --
        the markers are the only source.
        """
        totals = {inv.sequence_total for inv in result.invoices if inv.sequence_total}
        if result.declared_invoice_count is None and totals:
            result.declared_invoice_count = max(totals)
        if result.country_of_origin is None:
            for invoice in result.invoices:
                if invoice.supplier_country:
                    result.country_of_origin = invoice.supplier_country
                    break

    @staticmethod
    def _score_confidence(result: ParsedBillOfEntry) -> float:
        """Weighted field presence, in the shape the invoice parser uses.

        Weighted toward the fields a customs filing is actually keyed on: a
        BoE without its BE number is unusable however many goods lines came
        out, and one missing two of its seven invoices is worse than one that
        merely lost a port code.
        """
        score = 0
        if result.be_number:
            score += 15
        if result.be_date:
            score += 10
        if result.port_code:
            score += 5
        if result.iec or result.gstin:
            score += 10
        if result.invoices:
            score += 15
        if result.total_line_items:
            score += 15
        if result.duty.total_duty is not None:
            score += 10
        if result.duty.components:
            score += 5
        if not result.missing_invoice_sequences:
            score += 10
        if not result.validation_errors:
            score += 5
        return round(min(score, 100) / 100, 2)
