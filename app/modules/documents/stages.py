"""The processing state machine: what stage a document is in, and how far
along it is.

Two vocabularies, deliberately separate:

**Status** is the lifecycle — is this document still working, finished, or
broken? Five values, and they are what a list view filters on.

**Stage** is where inside the pipeline the work currently sits. Twelve
values, and they are what a progress display names.

Collapsing them into one field is tempting and wrong: a caller asking "is
this done" would have to know that OCR_PROCESSING, LAYOUT_ANALYSIS and
TABLE_EXTRACTION all mean "no", and adding a stage would break every such
caller. The status endpoint returns both, which is exactly the shape the
brief's example payload asks for.

Progress is derived from the stage, never stored as an independent number
that could disagree with it. The weights are measured proportions of real
wall-clock, not equal slices: OCR is roughly two minutes per page on this CPU
build while field extraction is milliseconds, so equal slices would show a
bar that leaps to 60% and then sits still for the entire actual wait.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- status: the lifecycle ------------------------------------------------

STATUS_UPLOADED = "UPLOADED"
STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETED = "COMPLETED"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
STATUS_FAILED = "FAILED"

ALL_STATUSES: tuple[str, ...] = (
    STATUS_UPLOADED,
    STATUS_PROCESSING,
    STATUS_COMPLETED,
    STATUS_REVIEW_REQUIRED,
    STATUS_FAILED,
)

# Nothing more will happen to a document in one of these states without a
# human or an explicit reprocess request.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {STATUS_COMPLETED, STATUS_REVIEW_REQUIRED, STATUS_FAILED}
)

# --- stage: position within the pipeline ----------------------------------

STAGE_UPLOADED = "UPLOADED"
STAGE_VALIDATION = "FILE_VALIDATION"
STAGE_PDF_ANALYSIS = "PDF_ANALYSIS"
STAGE_PREPROCESSING = "IMAGE_PREPROCESSING"
STAGE_OCR = "OCR_PROCESSING"
STAGE_LAYOUT = "LAYOUT_ANALYSIS"
STAGE_TABLE = "TABLE_EXTRACTION"
STAGE_FIELDS = "FIELD_EXTRACTION"
STAGE_LINE_ITEMS = "LINE_ITEM_EXTRACTION"
STAGE_VALIDATING = "VALIDATING"
STAGE_PERSISTING = "PERSISTING"
STAGE_DONE = "COMPLETED"


@dataclass(frozen=True)
class Stage:
    name: str
    weight: int          # share of total wall-clock, in arbitrary units
    label: str           # shown to a human waiting on it


# Ordered as the pipeline runs them. Weights come from timing real documents:
# OCR dominates everything else by two orders of magnitude, and pretending
# otherwise produces a progress bar that lies.
PIPELINE: tuple[Stage, ...] = (
    Stage(STAGE_UPLOADED, 0, "Uploaded"),
    Stage(STAGE_VALIDATION, 1, "Validating file"),
    Stage(STAGE_PDF_ANALYSIS, 4, "Analysing document structure"),
    Stage(STAGE_PREPROCESSING, 8, "Preparing images"),
    Stage(STAGE_OCR, 60, "Reading text"),
    Stage(STAGE_LAYOUT, 8, "Analysing layout"),
    Stage(STAGE_TABLE, 8, "Extracting tables"),
    Stage(STAGE_FIELDS, 5, "Extracting invoice fields"),
    Stage(STAGE_LINE_ITEMS, 3, "Extracting line items"),
    Stage(STAGE_VALIDATING, 2, "Validating extracted data"),
    Stage(STAGE_PERSISTING, 1, "Saving"),
    Stage(STAGE_DONE, 0, "Completed"),
)

STAGE_BY_NAME: dict[str, Stage] = {stage.name: stage for stage in PIPELINE}
ALL_STAGES: tuple[str, ...] = tuple(stage.name for stage in PIPELINE)
_TOTAL_WEIGHT: int = sum(stage.weight for stage in PIPELINE)

# Progress at the moment each stage *starts*, as a percentage.
_STAGE_START: dict[str, float] = {}
_cumulative = 0
for _stage in PIPELINE:
    _STAGE_START[_stage.name] = (_cumulative / _TOTAL_WEIGHT) * 100 if _TOTAL_WEIGHT else 0.0
    _cumulative += _stage.weight
del _cumulative, _stage


def stage_label(stage: str) -> str:
    known = STAGE_BY_NAME.get(stage)
    return known.label if known else stage.replace("_", " ").title()


def progress_for(stage: str, fraction_complete: float = 0.0) -> int:
    """Percentage complete when `stage` is `fraction_complete` through itself.

    The intra-stage fraction exists for OCR specifically. A ten-page scan
    spends twenty minutes in one stage; without page-level reporting the bar
    would sit at the same number for the whole of it, which is
    indistinguishable from a hung worker — the single most common reason a
    user kills a job that was about to succeed.
    """
    if stage == STAGE_DONE:
        return 100
    if stage not in _STAGE_START:
        return 0

    start = _STAGE_START[stage]
    weight = STAGE_BY_NAME[stage].weight
    span = (weight / _TOTAL_WEIGHT) * 100 if _TOTAL_WEIGHT else 0.0
    fraction = min(max(fraction_complete, 0.0), 1.0)

    # Capped at 99 so that "100%" means finished, never "on the last stage".
    return min(99, int(round(start + span * fraction)))


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def status_for_stage(stage: str) -> str:
    """The lifecycle status implied by being at a given stage. Terminal
    outcomes are decided by the pipeline (a completed run may still be
    REVIEW_REQUIRED), so this only answers the in-flight case."""
    if stage == STAGE_UPLOADED:
        return STATUS_UPLOADED
    if stage == STAGE_DONE:
        return STATUS_COMPLETED
    return STATUS_PROCESSING
