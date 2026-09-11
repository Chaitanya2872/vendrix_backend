"""API contracts for Bill of Entry extraction.

Pydantic models, separate from `dto.py` for the reason `invoices/dto.py`
gives: the dataclasses describe what the parser recovered, these describe
what a client is promised. They change for different reasons, and a field
the parser stops populating should be a visible contract change rather than
a silently-absent key.

The validation findings travel with the extraction rather than behind a
second endpoint. On this document class they are not an afterthought -- a
BoE whose IGST does not reconcile is exactly as "extracted" as one that does,
and a review screen that had to fetch them separately would render the
unreconciled one as finished.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

from app.modules.customs.dto import ParsedBillOfEntry


class BoeLineItemOut(BaseModel):
    serial: int | None = None
    cth: str | None = Field(default=None, examples=["82090090"],
                            description="Customs Tariff Head, eight digits.")
    description: str | None = None
    unit_price: Decimal | None = None
    quantity: Decimal | None = None
    uqc: str | None = Field(default=None, examples=["NOS"],
                            description="Unit Quantity Code.")
    amount: Decimal | None = None


class BoeInvoiceOut(BaseModel):
    sequence: str | None = Field(default=None, examples=["3/7"],
                                 description="Which Part-II page this is, as printed.")
    serial_no: int | None = None
    invoice_number: str | None = None
    invoice_date: date | None = None
    supplier_name: str | None = None
    supplier_address: str | None = None
    supplier_country: str | None = None
    invoice_value: Decimal | None = None
    currency: str | None = None
    incoterm: str | None = Field(default=None, examples=["CIF"])
    assessable_value: Decimal | None = None
    source_page: int | None = None
    line_items: list[BoeLineItemOut] = Field(default_factory=list)


class BoeDutyOut(BaseModel):
    """The assessed duty breakdown.

    Every head is nullable and none is defaulted to zero. A consignment
    attracts only some of them, and a zero here would assert an exemption
    that was never read off the document.
    """

    assessable_value: Decimal | None = None
    bcd: Decimal | None = Field(default=None, description="Basic customs duty.")
    acd: Decimal | None = None
    sws: Decimal | None = Field(default=None, description="Social welfare surcharge.")
    nccd: Decimal | None = None
    add_duty: Decimal | None = Field(default=None, description="Anti-dumping duty.")
    cvd: Decimal | None = None
    igst: Decimal | None = None
    gcess: Decimal | None = None
    total_duty: Decimal | None = None
    interest: Decimal | None = None
    penalty: Decimal | None = None
    fine: Decimal | None = None
    total_amount: Decimal | None = None


class BoeManifestOut(BaseModel):
    igm_number: str | None = None
    igm_date: date | None = None
    inward_date: date | None = None
    gigm_number: str | None = None
    gigm_date: date | None = None
    mawb_number: str | None = Field(default=None, description="Master air waybill.")
    mawb_date: date | None = None
    hawb_number: str | None = Field(default=None, description="House air waybill.")
    hawb_date: date | None = None
    packages: int | None = None
    gross_weight: Decimal | None = None


class BoeFinding(BaseModel):
    """One validation finding. Mirrors the invoice validator's shape so a
    review screen renders both document kinds through one component."""

    code: str = Field(examples=["boe_igst_amount_mismatch"])
    severity: str = Field(examples=["error", "warning"])
    message: str
    fields: list[str] = Field(default_factory=list)


class BillOfEntryOut(BaseModel):
    """Everything extracted from one Bill of Entry, plus how far to trust it."""

    be_number: str | None = Field(default=None, examples=["3560107"])
    be_date: date | None = None
    be_type: str | None = Field(default=None, examples=["Z"],
                                description="Z for an SEZ import.")
    port_code: str | None = Field(default=None, examples=["INURG6"])

    iec: str | None = None
    iec_branch: str | None = None
    gstin: str | None = None
    cb_code: str | None = Field(default=None, description="Customs broker licence.")
    ad_code: str | None = None

    importer_name: str | None = None
    importer_address: str | None = None
    cb_name: str | None = None

    country_of_origin: str | None = None
    country_of_consignment: str | None = None
    port_of_loading: str | None = None
    port_of_shipment: str | None = None

    declared_invoice_count: int | None = None
    declared_item_count: int | None = None
    declared_package_count: int | None = None
    gross_weight: Decimal | None = None

    exchange_rate: Decimal | None = None
    exchange_currency: str | None = None

    bond_number: str | None = None
    bond_port: str | None = None
    bond_type: str | None = None
    bond_debit_amount: Decimal | None = None

    authorised_person: str | None = None
    authorised_person_id: str | None = None

    duty: BoeDutyOut = Field(default_factory=BoeDutyOut)
    manifest: BoeManifestOut = Field(default_factory=BoeManifestOut)
    invoices: list[BoeInvoiceOut] = Field(default_factory=list)

    # --- provenance and trust ---
    parser_name: str
    parser_version: str
    used_ocr: bool
    page_count: int
    pages_parsed: int
    parsing_confidence: float = Field(ge=0.0, le=1.0)
    total_line_items: int
    missing_invoice_sequences: list[int] = Field(
        default_factory=list,
        description="Part-II pages the document declares but that were never "
                    "read -- normally the sign of a page-capped run. A "
                    "non-empty list means the extraction is incomplete, not "
                    "merely imperfect.",
    )
    duplicate_pages: list[int] = Field(
        default_factory=list,
        description="Scanned pages that repeated an already-read printed page "
                    "and were skipped.",
    )
    warnings: list[str] = Field(default_factory=list)
    findings: list[BoeFinding] = Field(default_factory=list)
    validation_clean: bool = Field(
        description="True when validation raised no errors. Warnings may "
                    "still be present.",
    )

    @classmethod
    def from_parsed(
        cls, parsed: ParsedBillOfEntry, report=None
    ) -> "BillOfEntryOut":
        """Build the response from a parse result and its validation report.

        `report` is optional so a caller that only wants the extraction does
        not have to run validation -- but the default of `validation_clean`
        is then False rather than True. Absent evidence of correctness is not
        evidence of correctness, and on a customs filing that distinction is
        the difference between a duty figure someone may rely on and one
        nobody has checked.
        """
        findings = [BoeFinding(**item.to_dict()) for item in report.findings] if report else []
        return cls(
            be_number=parsed.be_number,
            be_date=parsed.be_date,
            be_type=parsed.be_type,
            port_code=parsed.port_code,
            iec=parsed.iec,
            iec_branch=parsed.iec_branch,
            gstin=parsed.gstin,
            cb_code=parsed.cb_code,
            ad_code=parsed.ad_code,
            importer_name=parsed.importer_name,
            importer_address=parsed.importer_address,
            cb_name=parsed.cb_name,
            country_of_origin=parsed.country_of_origin,
            country_of_consignment=parsed.country_of_consignment,
            port_of_loading=parsed.port_of_loading,
            port_of_shipment=parsed.port_of_shipment,
            declared_invoice_count=parsed.declared_invoice_count,
            declared_item_count=parsed.declared_item_count,
            declared_package_count=parsed.declared_package_count,
            gross_weight=parsed.gross_weight,
            exchange_rate=parsed.exchange_rate,
            exchange_currency=parsed.exchange_currency,
            bond_number=parsed.bond_number,
            bond_port=parsed.bond_port,
            bond_type=parsed.bond_type,
            bond_debit_amount=parsed.bond_debit_amount,
            authorised_person=parsed.authorised_person,
            authorised_person_id=parsed.authorised_person_id,
            duty=BoeDutyOut(**vars(parsed.duty)),
            manifest=BoeManifestOut(**vars(parsed.manifest)),
            invoices=[
                BoeInvoiceOut(
                    sequence=invoice.sequence,
                    serial_no=invoice.serial_no,
                    invoice_number=invoice.invoice_number,
                    invoice_date=invoice.invoice_date,
                    supplier_name=invoice.supplier_name,
                    supplier_address=invoice.supplier_address,
                    supplier_country=invoice.supplier_country,
                    invoice_value=invoice.invoice_value,
                    currency=invoice.currency,
                    incoterm=invoice.incoterm,
                    assessable_value=invoice.assessable_value,
                    source_page=invoice.source_page,
                    line_items=[
                        BoeLineItemOut(
                            serial=item.serial,
                            cth=item.cth,
                            description=item.description,
                            unit_price=item.unit_price,
                            quantity=item.quantity,
                            uqc=item.uqc,
                            amount=item.amount,
                        )
                        for item in invoice.line_items
                    ],
                )
                for invoice in parsed.invoices
            ],
            parser_name=parsed.parser_name,
            parser_version=parsed.parser_version,
            used_ocr=parsed.used_ocr,
            page_count=parsed.page_count,
            pages_parsed=parsed.pages_parsed,
            parsing_confidence=parsed.parsing_confidence,
            total_line_items=parsed.total_line_items,
            missing_invoice_sequences=parsed.missing_invoice_sequences,
            duplicate_pages=parsed.duplicate_pages,
            warnings=parsed.warnings,
            findings=findings,
            validation_clean=bool(report.is_clean) if report else False,
        )
