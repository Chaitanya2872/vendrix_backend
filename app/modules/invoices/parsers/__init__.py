from .base_invoice_parser import BaseInvoiceParser
from .generic_invoice_parser import GenericInvoiceParser
from .ml_invoice_parser import MlAssistedInvoiceParser

# Ordered list: first parser whose can_parse() returns True wins. Vendor
# specific parsers should be inserted *before* GenericInvoiceParser so they
# get first refusal on documents they recognize.
#
# MlAssistedInvoiceParser runs the deterministic parser internally and then
# fills whatever it left blank, so it belongs first — but its can_parse()
# returns False whenever no trained artifact is present, which leaves the
# original GenericInvoiceParser behaviour untouched on an untrained checkout.
AVAILABLE_PARSERS: list[BaseInvoiceParser] = [
    MlAssistedInvoiceParser(),
    GenericInvoiceParser(),
]


def select_parser(text: str) -> BaseInvoiceParser:
    for parser in AVAILABLE_PARSERS:
        if parser.can_parse(text):
            return parser
    # GenericInvoiceParser.can_parse() only fails on empty text, but keep a
    # safe fallback so this never raises.
    return AVAILABLE_PARSERS[-1]
