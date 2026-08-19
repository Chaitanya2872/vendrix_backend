"""The golden corpus: invoices with known-correct answers.

Each entry is a layout, not just a document. The point of the set is that no
two entries are laid out the same way, because the claim the whole system
makes is that it does not need to know a vendor's template — and a corpus of
one layout repeated cannot test that claim at all.

**These are synthetic, and that is a real limitation.** They cover wording,
date format, number grouping, column order, party arrangement and page count,
but they come from one renderer and contain none of the noise a real scan
has. The moment real vendor invoices are available they should be dropped
into `tests/golden/real/` alongside a `.expected.json`, where the harness
picks them up automatically and weighs them the same. Real corrections from
the review screen are worth more than any number of these.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Checksum-valid GSTINs, so validation findings in the report come from the
# extraction rather than from placeholder numbers that were never going to
# validate.
KARNATAKA_A = "29AAGCB7383J1Z4"
KARNATAKA_B = "29AAECS1234K1Z9"
MAHARASHTRA = "27AAPFU0939F1ZV"
GUJARAT_A = "24AAACC1206D1ZM"
# A second Gujarat number: an intra-state invoice has two distinct
# parties in one state, and reusing one GSTIN for both is a data error
# the validator is right to reject.
GUJARAT_B = "24AAECS1234K1ZJ"


@dataclass
class GoldenInvoice:
    name: str
    description: str
    lines: list[tuple]                       # (text, x, y[, size])
    expected: dict
    pages: list[list[tuple]] = field(default_factory=list)  # multi-page override

    def write_pdf(self, directory: Path) -> Path:
        import fitz

        document = fitz.open()
        page_specs = self.pages or [self.lines]
        for spec in page_specs:
            page = document.new_page()
            for entry in spec:
                text, x, y = entry[0], entry[1], entry[2]
                size = entry[3] if len(entry) > 3 else 10
                # Monospaced so the column positions written here are the
                # column positions on the page.
                page.insert_text((x, y), text, fontsize=size, fontname="cour")
        path = directory / f"{self.name}.pdf"
        document.save(str(path))
        document.close()
        return path


def _standard_table(y0: int = 290) -> list[tuple]:
    return [
        ("Description        HSN     Qty      Rate       Amount", 50, y0),
        ("Steel Fabrication  7308     10   1500.00    15000.00", 50, y0 + 25),
        ("Welding Services   9988      4   1200.00     4800.00", 50, y0 + 45),
        ("Site Supervision   9983      2    500.00     1000.00", 50, y0 + 65),
    ]


GOLDEN_INVOICES: list[GoldenInvoice] = [
    GoldenInvoice(
        name="standard_intra_state",
        description="Common layout: labels left, values right, GST split CGST/SGST",
        lines=[
            ("ALPHA STEEL WORKS", 200, 60, 14),
            ("TAX INVOICE", 240, 85, 12),
            ("Invoice No: ASW/2026/0042", 50, 130),
            ("Invoice Date: 14/08/2026", 50, 150),
            ("Due Date: 13/09/2026", 50, 170),
            ("Vendor: Alpha Steel Works", 320, 130),
            (f"GSTIN: {KARNATAKA_A}", 320, 150),
            ("Bill To: Beta Constructions Pvt Ltd", 50, 210),
            (f"GSTIN: {KARNATAKA_B}", 50, 230),
            *_standard_table(),
            ("Sub Total                              20800.00", 250, 410),
            ("CGST 9%                                 1872.00", 250, 430),
            ("SGST 9%                                 1872.00", 250, 450),
            ("Grand Total                            24544.00", 250, 480),
        ],
        expected={
            "invoice_number": "ASW/2026/0042",
            "invoice_date": "2026-08-14",
            "due_date": "2026-09-13",
            "vendor_name": "Alpha Steel Works",
            "vendor_gstin": KARNATAKA_A,
            "customer_name": "Beta Constructions Pvt Ltd",
            "customer_gstin": KARNATAKA_B,
            "subtotal": 20800.00,
            "cgst_amount": 1872.00,
            "sgst_amount": 1872.00,
            "total_amount": 24544.00,
            "line_item_count": 3,
        },
    ),

    GoldenInvoice(
        name="inter_state_igst",
        description="Different states, so IGST rather than CGST/SGST; alternate wordings",
        lines=[
            ("GAMMA ENGINEERING", 200, 60, 14),
            ("Bill No: GE-2026-117", 50, 130),
            ("Bill Date: 03/04/2026", 50, 150),
            ("Supplier: Gamma Engineering", 320, 130),
            (f"GST No: {KARNATAKA_A}", 320, 150),
            ("Buyer: Delta Infra Limited", 50, 210),
            (f"GST No: {MAHARASHTRA}", 50, 230),
            ("Particulars        SAC     Quantity  Price      Total", 50, 290),
            ("Design Consulting  9983      20   2500.00    50000.00", 50, 315),
            ("Site Survey        9983       5   1000.00     5000.00", 50, 335),
            ("Taxable Value                          55000.00", 250, 400),
            ("IGST 18%                                9900.00", 250, 420),
            ("Amount Payable                         64900.00", 250, 450),
        ],
        expected={
            "invoice_number": "GE-2026-117",
            # Day-first: 03/04 is 3 April, not 4 March. India-only deployment.
            "invoice_date": "2026-04-03",
            "vendor_gstin": KARNATAKA_A,
            "customer_gstin": MAHARASHTRA,
            "taxable_amount": 55000.00,
            "igst_amount": 9900.00,
            "total_amount": 64900.00,
            "line_item_count": 2,
        },
    ),

    GoldenInvoice(
        name="lakh_grouping_and_month_names",
        description="Indian lakh grouping, month-name date, round-off present",
        lines=[
            ("OMEGA TRADERS", 200, 60, 14),
            ("Tax Invoice No: OT/26-27/0891", 50, 130),
            ("Dated: 14 Aug 2026", 50, 150),
            ("Sold By: Omega Traders", 320, 130),
            (f"GSTIN: {GUJARAT_A}", 320, 150),
            ("Sold To: Sigma Retail LLP", 50, 210),
            (f"GSTIN: {GUJARAT_B}", 50, 230),
            ("Item Description   HSN Code  Qty   Unit Price  Net Amount", 50, 290),
            ("Cement Bags        2523     500     380.00    190000.00", 50, 315),
            ("Sub Total                            190000.00", 250, 390),
            ("CGST 9%                               17100.00", 250, 410),
            ("SGST 9%                               17100.00", 250, 430),
            ("Round Off                                 0.04", 250, 450),
            ("Grand Total                        2,24,200.04", 250, 480),
        ],
        expected={
            "invoice_number": "OT/26-27/0891",
            "invoice_date": "2026-08-14",
            "subtotal": 190000.00,
            "cgst_amount": 17100.00,
            "sgst_amount": 17100.00,
            "round_off": 0.04,
            "total_amount": 224200.04,
            "line_item_count": 1,
        },
    ),

    GoldenInvoice(
        name="buyer_block_first",
        description="Buyer above seller — reading order is not evidence of who is who",
        lines=[
            ("TAX INVOICE", 240, 60, 12),
            ("Document No: ZR-99120", 50, 110),
            ("Date of Invoice: 2026-08-14", 50, 130),
            ("Billed To: Zeta Motors Pvt Ltd", 50, 180),
            (f"GSTIN: {KARNATAKA_B}", 50, 200),
            ("Issued By: Theta Components", 50, 250),
            (f"GSTIN: {KARNATAKA_A}", 50, 270),
            *_standard_table(320),
            ("Sub Total                              20800.00", 250, 440),
            ("Total Tax                               3744.00", 250, 460),
            ("Net Payable                            24544.00", 250, 490),
        ],
        expected={
            "invoice_number": "ZR-99120",
            "invoice_date": "2026-08-14",
            "vendor_gstin": KARNATAKA_A,
            "customer_gstin": KARNATAKA_B,
            "subtotal": 20800.00,
            "tax_amount": 3744.00,
            "total_amount": 24544.00,
            "line_item_count": 3,
        },
    ),

    GoldenInvoice(
        name="two_page_totals_overleaf",
        description="Line items on page one, totals on page two",
        lines=[],
        pages=[
            [
                ("KAPPA SUPPLIES", 200, 60, 14),
                ("Invoice Number: KS-2026-0007", 50, 120),
                ("Invoice Date: 14/08/2026", 50, 140),
                ("Vendor: Kappa Supplies", 320, 120),
                (f"GSTIN: {KARNATAKA_A}", 320, 140),
                ("Bill To: Lambda Works", 50, 190),
                (f"GSTIN: {KARNATAKA_B}", 50, 210),
                *_standard_table(260),
            ],
            [
                ("Continued from page 1", 50, 60),
                ("Sub Total                              20800.00", 250, 200),
                ("CGST 9%                                 1872.00", 250, 220),
                ("SGST 9%                                 1872.00", 250, 240),
                ("Grand Total                            24544.00", 250, 270),
            ],
        ],
        expected={
            "invoice_number": "KS-2026-0007",
            "invoice_date": "2026-08-14",
            "vendor_gstin": KARNATAKA_A,
            "customer_gstin": KARNATAKA_B,
            "subtotal": 20800.00,
            # The value the system must not take from page one instead.
            "total_amount": 24544.00,
            "line_item_count": 3,
        },
    ),

    GoldenInvoice(
        name="summary_only_no_table",
        description="No line-item table at all — a service invoice",
        lines=[
            ("MU CONSULTING", 200, 60, 14),
            ("Invoice #: MC-2026-45", 50, 130),
            ("Issue Date: 14/08/2026", 50, 150),
            ("Payment Due Date: 28/08/2026", 50, 170),
            ("Seller: Mu Consulting", 320, 130),
            (f"GSTIN: {KARNATAKA_A}", 320, 150),
            ("Customer: Nu Holdings", 50, 210),
            (f"GSTIN: {KARNATAKA_B}", 50, 230),
            ("Professional fees for the month of July 2026", 50, 290),
            ("Amount Before Tax                      80000.00", 250, 360),
            ("CGST 9%                                 7200.00", 250, 380),
            ("SGST 9%                                 7200.00", 250, 400),
            ("Total Amount Payable                   94400.00", 250, 430),
        ],
        expected={
            "invoice_number": "MC-2026-45",
            "invoice_date": "2026-08-14",
            "due_date": "2026-08-28",
            "subtotal": 80000.00,
            "cgst_amount": 7200.00,
            "sgst_amount": 7200.00,
            "total_amount": 94400.00,
            "line_item_count": 0,
        },
    ),
]


def build_corpus(directory: Path) -> list[tuple[Path, dict]]:
    """Write every golden invoice and return (path, expected) pairs."""
    directory.mkdir(parents=True, exist_ok=True)
    return [(invoice.write_pdf(directory), invoice.expected) for invoice in GOLDEN_INVOICES]


def load_real_corpus(directory: Path) -> list[tuple[Path, dict]]:
    """Pick up real vendor invoices dropped alongside their expected JSON.

    Nothing here needs registering: a `foo.pdf` next to a `foo.expected.json`
    joins the set. Keeping the barrier that low is deliberate — the corpus is
    only worth anything if adding a real invoice to it takes a minute.
    """
    if not directory.is_dir():
        return []

    found: list[tuple[Path, dict]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        expectation = path.with_suffix(".expected.json")
        if not expectation.exists():
            continue
        found.append((path, json.loads(expectation.read_text(encoding="utf-8"))))
    return found
