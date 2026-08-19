"""Synthetic GST invoice generator with exact ground truth.

Training a field-extraction model needs labelled documents, and hand-labelling
a corpus large enough to cover seven file formats is not realistic here. So we
go the other way: generate the invoice *content* first — which makes the ground
truth free and exact — then render it into every format and push it through the
real extraction pipeline. The labels then come from matching the recovered text
back to the values we know we wrote (see corpus.py).

The generator's job is therefore variation, not realism-for-its-own-sake. A
model trained on one label wording and one date format learns that wording and
that format; it would score well here and fail on the first real invoice. So
every surface detail a real invoice varies — label phrasing, date format,
currency marker, digit grouping, block order, intra- vs inter-state tax
breakdown — is randomised per document.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP

_CENTS = Decimal("0.01")
RUPEE = "₹"

_COMPANY_HEADS = (
    "Aravind", "Bharat", "Chetan", "Deccan", "Everest", "Ganga", "Himalaya",
    "Indus", "Kaveri", "Lotus", "Meridian", "Narmada", "Orbit", "Pinnacle",
    "Quantum", "Rashtra", "Sahyadri", "Trident", "Uttara", "Vindhya", "Zenith",
    "Nilgiri", "Konkan", "Malabar", "Coromandel", "Saraswati", "Tapti",
)
_COMPANY_TAILS = (
    "Logistics Pvt Ltd", "Industries Limited", "Technologies Pvt Ltd",
    "Enterprises", "Traders", "Solutions LLP", "Motors Pvt Ltd",
    "Steel Works", "Packaging Pvt Ltd", "Agro Exports", "Systems India Pvt Ltd",
    "Infrastructure Ltd", "Chemicals Pvt Ltd", "Textile Mills Ltd",
)
_CITIES = (
    ("Bengaluru", "Karnataka", "560001", "29"),
    ("Mumbai", "Maharashtra", "400001", "27"),
    ("Chennai", "Tamil Nadu", "600001", "33"),
    ("Hyderabad", "Telangana", "500001", "36"),
    ("Pune", "Maharashtra", "411001", "27"),
    ("Ahmedabad", "Gujarat", "380001", "24"),
    ("New Delhi", "Delhi", "110001", "07"),
    ("Kolkata", "West Bengal", "700001", "19"),
    ("Kochi", "Kerala", "682001", "32"),
    ("Jaipur", "Rajasthan", "302001", "08"),
)
_STREETS = (
    "Industrial Layout", "MG Road", "Ring Road", "Sector 21", "Phase II",
    "Export Promotion Park", "Trade Centre", "Logistics Park", "GIDC Estate",
)
_GOODS = (
    ("Steel fasteners M12", "73181500", "Nos"),
    ("HDPE packaging film", "39201019", "Kg"),
    ("Freight charges Bengaluru to Pune", "996511", "Trip"),
    ("Cotton yarn 40s", "52051110", "Kg"),
    ("Diesel generator service", "998719", "Hrs"),
    ("Industrial lubricant SAE 40", "27101980", "Ltr"),
    ("Corrugated boxes 5-ply", "48191010", "Nos"),
    ("Warehouse handling charges", "996729", "MT"),
    ("Cloud application development", "998314", "Hrs"),
    ("Annual maintenance contract", "998719", "Nos"),
    ("Tyre retreading service", "998729", "Nos"),
    ("Safety helmets ISI marked", "65061010", "Nos"),
    ("Aluminium extrusion sections", "76042990", "Kg"),
    ("Courier and last-mile delivery", "996812", "Shpt"),
)

# Label wording alternatives. Real invoices phrase the same field a dozen
# ways; the model must key off the shape of the value plus *some* nearby
# label, not one exact string.
LABEL_WORDINGS: dict[str, tuple[str, ...]] = {
    "invoice_number": ("Invoice No.", "Invoice Number", "Invoice #", "Bill No.", "Tax Invoice No.", "Inv No"),
    "invoice_date": ("Invoice Date", "Date", "Dated", "Bill Date", "Date of Issue"),
    "due_date": ("Due Date", "Payment Due", "Due On", "Payment Due Date"),
    "vendor_block": ("Seller", "Vendor", "From", "Bill From", "Supplier", "Sold By"),
    "customer_block": ("Buyer", "Bill To", "Customer", "Billed To", "Consignee"),
    "gstin": ("GSTIN", "GSTIN/UIN", "GST No.", "GSTIN No"),
    "subtotal": ("Subtotal", "Sub Total", "Taxable Value", "Amount before Tax", "Total Taxable Value"),
    "cgst": ("CGST", "Central GST", "CGST Amount"),
    "sgst": ("SGST", "State GST", "SGST Amount"),
    "igst": ("IGST", "Integrated GST", "IGST Amount"),
    "tax": ("Total Tax", "Tax Amount", "Total GST", "GST Total"),
    "total": ("Total", "Grand Total", "Total Amount", "Invoice Total", "Amount Payable", "Total Payable"),
}

DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%Y-%m-%d", "%d.%m.%Y", "%b %d, %Y")
CURRENCY_MARKERS = ("Rs. ", "INR ", RUPEE, "")


@dataclass
class SynthLineItem:
    description: str
    hsn_sac: str
    quantity: Decimal
    unit: str
    unit_price: Decimal
    taxable_value: Decimal


@dataclass
class SynthInvoice:
    """One generated invoice: the values, plus the presentation choices used
    to render it. Renderers read both — the ground truth is the values, and
    the presentation choices are what make two documents with identical
    values look nothing alike."""

    doc_id: str
    invoice_number: str
    invoice_date: date
    due_date: date

    vendor_name: str
    vendor_gstin: str
    vendor_address: list[str]
    customer_name: str
    customer_gstin: str
    customer_address: list[str]

    line_items: list[SynthLineItem]
    subtotal: Decimal
    cgst_amount: Decimal | None
    sgst_amount: Decimal | None
    igst_amount: Decimal | None
    tax_amount: Decimal
    total_amount: Decimal
    gst_rate: Decimal
    interstate: bool

    wording: dict[str, str]
    date_format: str
    currency: str
    indian_grouping: bool
    show_due_date: bool
    show_tax_total: bool
    parties_side_by_side: bool

    def formatted_date(self, value: date) -> str:
        return value.strftime(self.date_format)

    def money(self, value: Decimal) -> str:
        """Render an amount the way this document renders amounts — including
        Indian lakh grouping, which naive thousands-separator handling gets
        wrong (1,25,000.00 rather than 125,000.00)."""
        return f"{self.currency}{group_digits(value, self.indian_grouping)}"

    def ground_truth(self) -> dict[str, str]:
        """Label -> the exact string this document should yield for it.

        Money and date labels are compared numerically downstream, so the
        rendered form here is only a starting point for matching; names and
        identifiers are compared as text.
        """
        truth = {
            "INVOICE_NUMBER": self.invoice_number,
            "INVOICE_DATE": self.formatted_date(self.invoice_date),
            "VENDOR_NAME": self.vendor_name,
            "VENDOR_GSTIN": self.vendor_gstin,
            "CUSTOMER_NAME": self.customer_name,
            "CUSTOMER_GSTIN": self.customer_gstin,
            "SUBTOTAL": self.money(self.subtotal),
            "TOTAL_AMOUNT": self.money(self.total_amount),
        }
        if self.show_due_date:
            truth["DUE_DATE"] = self.formatted_date(self.due_date)
        if self.interstate:
            truth["IGST_AMOUNT"] = self.money(self.igst_amount)
        else:
            truth["CGST_AMOUNT"] = self.money(self.cgst_amount)
            truth["SGST_AMOUNT"] = self.money(self.sgst_amount)
        if self.show_tax_total:
            truth["TAX_AMOUNT"] = self.money(self.tax_amount)
        return truth

    def numeric_truth(self) -> dict[str, Decimal]:
        """Money labels as Decimals, for tolerance-based comparison that
        survives OCR mangling a currency symbol or a thousands separator."""
        values = {"SUBTOTAL": self.subtotal, "TOTAL_AMOUNT": self.total_amount}
        if self.interstate:
            values["IGST_AMOUNT"] = self.igst_amount
        else:
            values["CGST_AMOUNT"] = self.cgst_amount
            values["SGST_AMOUNT"] = self.sgst_amount
        if self.show_tax_total:
            values["TAX_AMOUNT"] = self.tax_amount
        return values

    def date_truth(self) -> dict[str, date]:
        values = {"INVOICE_DATE": self.invoice_date}
        if self.show_due_date:
            values["DUE_DATE"] = self.due_date
        return values


def group_digits(value: Decimal, indian: bool) -> str:
    """Thousands grouping. Indian grouping puts the first separator after the
    last three digits and every two digits thereafter."""
    quantised = value.quantize(_CENTS, rounding=ROUND_HALF_UP)
    sign = "-" if quantised < 0 else ""
    whole, _, fraction = str(abs(quantised)).partition(".")
    fraction = (fraction + "00")[:2]
    if not indian:
        return f"{sign}{int(whole):,}.{fraction}"
    if len(whole) <= 3:
        return f"{sign}{whole}.{fraction}"
    head, tail = whole[:-3], whole[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return f"{sign}{','.join(parts)},{tail}.{fraction}"


def _gstin(rng: random.Random, state_code: str) -> str:
    """A structurally valid GSTIN: state code + PAN + entity code + 'Z' +
    checksum char. Matches the pattern gst_utils.GSTIN_PATTERN enforces, so
    generated documents exercise the same extraction path real invoices do."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pan = (
        "".join(rng.choice(letters) for _ in range(5))
        + f"{rng.randint(0, 9999):04d}"
        + rng.choice(letters)
    )
    entity = rng.choice("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    checksum = rng.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"{state_code}{pan}{entity}Z{checksum}"


def _company(rng: random.Random, state_code: str | None = None) -> tuple[str, list[str], str]:
    """Pick a company. `state_code` constrains the city choice, which is how
    the caller controls whether an invoice ends up intra-state (CGST + SGST)
    or inter-state (IGST)."""
    name = f"{rng.choice(_COMPANY_HEADS)} {rng.choice(_COMPANY_TAILS)}"
    pool = [entry for entry in _CITIES if state_code is None or entry[3] == state_code]
    city, state, pin, chosen_code = rng.choice(pool or list(_CITIES))
    address = [
        f"{rng.randint(1, 480)}, {rng.choice(_STREETS)}",
        f"{city}, {state} {pin}",
    ]
    return name, address, chosen_code


def generate_invoice(seed: int) -> SynthInvoice:
    """Build one invoice deterministically from `seed`, so a corpus can be
    regenerated exactly — and a document that trips the pipeline reproduced —
    from its id alone."""
    rng = random.Random(seed)

    vendor_name, vendor_address, vendor_state = _company(rng)
    # Choose the tax treatment first, then a customer city that produces it.
    # Left to chance over a 10-city pool, intra-state invoices would be ~12%
    # of the corpus and the CGST/SGST classes would be too rare to learn —
    # an artefact of the generator, not of how invoices actually arrive.
    want_interstate = rng.random() < 0.55
    if want_interstate:
        customer_name, customer_address, customer_state = _company(rng)
        while customer_state == vendor_state:
            customer_name, customer_address, customer_state = _company(rng)
    else:
        customer_name, customer_address, customer_state = _company(rng, state_code=vendor_state)
    interstate = vendor_state != customer_state

    issued = date(2024, 1, 1) + timedelta(days=rng.randint(0, 900))
    due = issued + timedelta(days=rng.choice((7, 15, 30, 45, 60)))

    gst_rate = Decimal(rng.choice((5, 12, 18, 28)))
    items: list[SynthLineItem] = []
    for description, hsn, unit in rng.sample(_GOODS, rng.randint(2, 6)):
        quantity = Decimal(rng.randint(1, 40))
        unit_price = (
            Decimal(rng.randint(50, 24000)) + Decimal(rng.choice(("0.00", "0.50", "0.75")))
        ).quantize(_CENTS)
        items.append(
            SynthLineItem(
                description=description,
                hsn_sac=hsn,
                quantity=quantity,
                unit=unit,
                unit_price=unit_price,
                taxable_value=(quantity * unit_price).quantize(_CENTS, rounding=ROUND_HALF_UP),
            )
        )

    subtotal = sum((item.taxable_value for item in items), Decimal("0")).quantize(_CENTS)
    tax_amount = (subtotal * gst_rate / Decimal("100")).quantize(_CENTS, rounding=ROUND_HALF_UP)
    if interstate:
        igst, cgst, sgst = tax_amount, None, None
    else:
        cgst = (tax_amount / 2).quantize(_CENTS, rounding=ROUND_HALF_UP)
        sgst = (tax_amount - cgst).quantize(_CENTS)
        igst = None
    total = (subtotal + tax_amount).quantize(_CENTS)

    prefix = rng.choice(("INV", "TI", "GST", "SI", "BILL", "INV/23-24"))
    invoice_number = f"{prefix}{rng.choice(('-', '/', ''))}{rng.randint(1, 99999):05d}"

    return SynthInvoice(
        doc_id=f"synth-{seed:06d}",
        invoice_number=invoice_number,
        invoice_date=issued,
        due_date=due,
        vendor_name=vendor_name,
        vendor_gstin=_gstin(rng, vendor_state),
        vendor_address=vendor_address,
        customer_name=customer_name,
        customer_gstin=_gstin(rng, customer_state),
        customer_address=customer_address,
        line_items=items,
        subtotal=subtotal,
        cgst_amount=cgst,
        sgst_amount=sgst,
        igst_amount=igst,
        tax_amount=tax_amount,
        total_amount=total,
        gst_rate=gst_rate,
        interstate=interstate,
        wording={key: rng.choice(options) for key, options in LABEL_WORDINGS.items()},
        date_format=rng.choice(DATE_FORMATS),
        currency=rng.choice(CURRENCY_MARKERS),
        indian_grouping=rng.random() < 0.6,
        show_due_date=rng.random() < 0.75,
        show_tax_total=rng.random() < 0.7,
        parties_side_by_side=rng.random() < 0.5,
    )


def generate_corpus(count: int, seed_offset: int = 0) -> list[SynthInvoice]:
    return [generate_invoice(seed_offset + index) for index in range(count)]
