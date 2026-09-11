"""Customs documents: the Bill of Entry parsing pipeline.

Additive by design. Nothing in `app.modules.invoices` is modified to support
this document type — a Bill of Entry has its own DTO, its own label lexicon
and its own validators, because the fields it shares with an invoice mostly
do not mean the same thing (see the note in `dto.py`).
"""
