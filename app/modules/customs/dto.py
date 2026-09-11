"""Internal data-transfer objects for the Bill of Entry parsing pipeline.

Plain dataclasses, for the same reason `invoices/dto.py` uses them: these
describe *what the parser recovered*, not what any API returns, and keeping
the two apart stops a change to one silently redefining the other.

**Why a Bill of Entry is not an Invoice.** It is tempting to reuse
`ParsedInvoiceResult` -- a BoE has invoice numbers, amounts and line items,
so the shapes rhyme. They are not the same document:

  * One BoE carries *many* commercial invoices (the sample consignment
    declares seven), each with its own supplier, currency and line items.
    An invoice carries one.
  * Its money is a statutory duty breakdown (BCD, SWS, IGST, cess), not a
    GST breakdown. ``igst_amount`` means "tax charged by a seller" on an
    invoice and "integrated tax assessed on import" here; collapsing them
    would make downstream totals wrong in a way nothing would flag.
  * Its authoritative fields -- BE number, port code, bond, manifest -- have
    no invoice counterpart at all.

So the BoE gets its own DTO, and `customs/validation.py` its own checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


@dataclass
class BoeDuty:
    """Part-I section C: the assessed duty breakdown.

    Every component is optional because a consignment attracts only some of
    them -- the sample pays BCD, SWS and IGST and nothing else, and a parser
    that invented a zero for the rest would be asserting an exemption it
    never read.
    """

    assessable_value: Decimal | None = None   # 18.TOT ASS VAL
    bcd: Decimal | None = None                # 1.BCD   basic customs duty
    acd: Decimal | None = None                # 2.ACD
    sws: Decimal | None = None                # 3.SWS   social welfare surcharge
    nccd: Decimal | None = None               # 4.NCCD
    add_duty: Decimal | None = None           # 5.ADD   anti-dumping duty
    cvd: Decimal | None = None                # 6.CVD
    igst: Decimal | None = None               # 7.IGST
    gcess: Decimal | None = None              # 8.G.CESS
    total_duty: Decimal | None = None         # 14.TOTAL DUTY
    interest: Decimal | None = None           # 15.INT
    penalty: Decimal | None = None            # 16.PNLTY
    fine: Decimal | None = None               # 17.FINE
    total_amount: Decimal | None = None       # 19.TOT.AMOUNT

    @property
    def components(self) -> list[Decimal]:
        """The duty heads that make up TOTAL DUTY, skipping the unread ones."""
        return [
            value
            for value in (
                self.bcd, self.acd, self.sws, self.nccd,
                self.add_duty, self.cvd, self.igst, self.gcess,
            )
            if value is not None
        ]


@dataclass
class BoeManifest:
    """Part-I section D: how the goods physically arrived."""

    igm_number: str | None = None
    igm_date: date | None = None
    inward_date: date | None = None
    gigm_number: str | None = None
    gigm_date: date | None = None
    mawb_number: str | None = None
    mawb_date: date | None = None
    hawb_number: str | None = None
    hawb_date: date | None = None
    packages: int | None = None
    gross_weight: Decimal | None = None


@dataclass
class BoeLineItem:
    """One goods line inside one Part-II invoice."""

    serial: int | None = None
    cth: str | None = None                # customs tariff head, 8 digits
    description: str | None = None
    unit_price: Decimal | None = None
    quantity: Decimal | None = None
    uqc: str | None = None                # unit quantity code, e.g. NOS
    amount: Decimal | None = None
    raw_lines: list[str] = field(default_factory=list)


@dataclass
class BoeInvoiceRef:
    """One commercial invoice declared on the BoE (one Part-II page).

    ``sequence_index``/``sequence_total`` come from the "3/7" marker printed
    in the Part-II banner. It is the only reliable way to tell which invoice
    a page belongs to and -- more importantly -- to know which pages are
    *missing*; see `ParsedBillOfEntry.missing_invoice_sequences`.
    """

    sequence_index: int | None = None         # the 3 in "3/7"
    sequence_total: int | None = None         # the 7 in "3/7"
    serial_no: int | None = None              # 1.S.NO
    invoice_number: str | None = None         # 2.INVOICE NO
    invoice_date: date | None = None          # 2. ... & DT
    supplier_name: str | None = None          # 3.SUPPLIER NAME
    supplier_address: str | None = None
    supplier_country: str | None = None
    invoice_value: Decimal | None = None      # 1.INV VALUE
    currency: str | None = None               # 14.Cur
    incoterm: str | None = None               # 15.Term, e.g. CIF
    assessable_value: Decimal | None = None   # 14.ASS. VALUE
    source_page: int | None = None            # 1-based page of the PDF
    line_items: list[BoeLineItem] = field(default_factory=list)

    @property
    def sequence(self) -> str | None:
        if self.sequence_index is None or self.sequence_total is None:
            return None
        return f"{self.sequence_index}/{self.sequence_total}"


@dataclass
class ParsedBillOfEntry:
    """Everything recovered from one Bill of Entry document."""

    # --- Part-I header ---
    be_number: str | None = None
    be_date: date | None = None
    be_type: str | None = None                # Z for SEZ
    port_code: str | None = None
    iec: str | None = None
    iec_branch: str | None = None
    gstin: str | None = None
    cb_code: str | None = None                # customs broker code
    ad_code: str | None = None

    importer_name: str | None = None
    importer_address: str | None = None
    cb_name: str | None = None

    country_of_origin: str | None = None
    country_of_consignment: str | None = None
    port_of_loading: str | None = None
    port_of_shipment: str | None = None

    # Declared counts from the header strip. Cross-checked against what was
    # actually parsed -- the mismatch is the whole reason to keep them.
    declared_invoice_count: int | None = None
    declared_item_count: int | None = None
    declared_container_count: int | None = None
    declared_package_count: int | None = None
    gross_weight: Decimal | None = None

    exchange_rate: Decimal | None = None
    exchange_currency: str | None = None

    # --- sections ---
    duty: BoeDuty = field(default_factory=BoeDuty)
    manifest: BoeManifest = field(default_factory=BoeManifest)

    bond_number: str | None = None
    bond_port: str | None = None
    bond_type: str | None = None
    bond_debit_amount: Decimal | None = None

    authorised_person: str | None = None
    authorised_person_id: str | None = None

    invoices: list[BoeInvoiceRef] = field(default_factory=list)

    # --- provenance ---
    parser_name: str = "bill_of_entry"
    parser_version: str = "1.0"
    used_ocr: bool = False
    page_count: int = 0
    pages_parsed: int = 0
    duplicate_pages: list[int] = field(default_factory=list)
    parsing_confidence: float = 0.0

    warnings: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def total_line_items(self) -> int:
        return sum(len(invoice.line_items) for invoice in self.invoices)

    @property
    def missing_invoice_sequences(self) -> list[int]:
        """Which "n/7" pages were declared but never seen.

        This is the check that catches a truncated read. ``ocr_max_pages``
        defaults to 3, and a seven-invoice BoE puts invoices 3 through 7 on
        pages 4 onward -- so a silently capped run yields a perfectly
        well-formed result missing most of the goods. Nothing else in the
        pipeline would notice.
        """
        totals = {inv.sequence_total for inv in self.invoices if inv.sequence_total}
        expected = max(totals) if totals else self.declared_invoice_count
        if not expected:
            return []
        seen = {inv.sequence_index for inv in self.invoices if inv.sequence_index}
        return [n for n in range(1, expected + 1) if n not in seen]

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def add_validation_error(self, message: str) -> None:
        if message not in self.validation_errors:
            self.validation_errors.append(message)
