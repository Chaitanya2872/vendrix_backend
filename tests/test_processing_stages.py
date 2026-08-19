"""The processing state machine: stage vocabulary, progress arithmetic, and
the tracker that publishes both.

Progress is the part users judge the system by. A bar that jumps to 60% and
freezes for twenty minutes reads as a hung job, and a killed job that was
about to succeed costs more than a slow one.
"""
import pytest

from app.modules.documents import stages
from app.modules.documents.stages import (
    ALL_STAGES,
    PIPELINE,
    STAGE_DONE,
    STAGE_OCR,
    STAGE_TABLE,
    STAGE_UPLOADED,
    STAGE_VALIDATION,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    STATUS_REVIEW_REQUIRED,
    is_terminal,
    progress_for,
    stage_label,
    status_for_stage,
)


class TestVocabulary:
    def test_every_stage_the_brief_names_exists(self):
        for name in (
            "OCR_PROCESSING", "LAYOUT_ANALYSIS", "TABLE_EXTRACTION",
            "FIELD_EXTRACTION", "VALIDATING", "COMPLETED",
        ):
            assert name in ALL_STAGES

    def test_every_status_the_brief_names_exists(self):
        for name in ("UPLOADED", "PROCESSING", "COMPLETED", "REVIEW_REQUIRED", "FAILED"):
            assert name in stages.ALL_STATUSES

    def test_stage_names_are_unique(self):
        assert len(ALL_STAGES) == len(set(ALL_STAGES))

    def test_every_stage_has_a_human_label(self):
        for stage in PIPELINE:
            assert stage.label and stage.label != stage.name

    def test_an_unknown_stage_still_produces_a_readable_label(self):
        assert stage_label("SOME_NEW_STAGE") == "Some New Stage"


class TestProgress:
    def test_progress_never_goes_backwards_across_the_pipeline(self):
        values = [progress_for(stage.name) for stage in PIPELINE]
        assert values == sorted(values)

    def test_the_first_stage_starts_at_zero_and_completion_is_a_hundred(self):
        assert progress_for(STAGE_UPLOADED) == 0
        assert progress_for(STAGE_DONE) == 100

    def test_no_in_flight_stage_ever_reports_a_hundred(self):
        # 100% must mean finished. A stage reporting 100 while still working
        # is how a client concludes it can stop polling.
        for stage in PIPELINE:
            if stage.name != STAGE_DONE:
                assert progress_for(stage.name, fraction_complete=1.0) < 100

    def test_ocr_occupies_most_of_the_bar_because_it_occupies_most_of_the_time(self):
        ocr_span = progress_for(STAGE_TABLE) - progress_for(STAGE_OCR)
        assert ocr_span > 50, "OCR is minutes per page; the bar must reflect that"

    def test_intra_stage_progress_moves_within_the_stage_only(self):
        start = progress_for(STAGE_OCR, 0.0)
        middle = progress_for(STAGE_OCR, 0.5)
        end = progress_for(STAGE_OCR, 1.0)
        next_stage = progress_for(stages.STAGE_LAYOUT)
        assert start < middle < end <= next_stage

    def test_a_ten_page_ocr_job_reports_ten_distinct_values(self):
        # The whole reason substep exists: without it a ten-page scan sits on
        # one number for twenty minutes.
        values = {progress_for(STAGE_OCR, page / 10) for page in range(10)}
        assert len(values) >= 8

    def test_out_of_range_fractions_are_clamped_not_extrapolated(self):
        assert progress_for(STAGE_OCR, -5) == progress_for(STAGE_OCR, 0.0)
        assert progress_for(STAGE_OCR, 99) == progress_for(STAGE_OCR, 1.0)

    def test_an_unknown_stage_reports_zero_rather_than_raising(self):
        assert progress_for("NOT_A_STAGE") == 0


class TestStatusMapping:
    def test_terminal_statuses_are_exactly_the_ones_needing_no_further_work(self):
        assert is_terminal(STATUS_COMPLETED)
        assert is_terminal(STATUS_REVIEW_REQUIRED)
        assert is_terminal(STATUS_FAILED)
        assert not is_terminal(STATUS_PROCESSING)

    def test_an_in_flight_stage_maps_to_processing(self):
        assert status_for_stage(STAGE_OCR) == STATUS_PROCESSING
        assert status_for_stage(STAGE_VALIDATION) == STATUS_PROCESSING

    def test_the_boundary_stages_map_to_their_own_statuses(self):
        assert status_for_stage(STAGE_UPLOADED) == "UPLOADED"
        assert status_for_stage(STAGE_DONE) == STATUS_COMPLETED


class TestTracker:
    @pytest.fixture
    def tracker(self, db, seeded_document):
        from app.modules.documents import repository
        from app.modules.documents.service import ProcessingTracker

        run = repository.create_run(db, seeded_document.id)
        db.commit()
        return ProcessingTracker(db, run, document_number=seeded_document.document_number)

    def test_progress_is_visible_to_a_separate_session_as_soon_as_it_is_published(self, tracker):
        """A poller reads through its own session. Progress batched into the
        job's final commit would be invisible for the entire wait."""
        from app.db.session import SessionLocal
        from app.modules.documents.models import DocumentProcessingRun

        tracker.stage(STAGE_OCR)

        with SessionLocal() as observer:
            observed = observer.get(DocumentProcessingRun, tracker.run.id)
            assert observed.current_stage == STAGE_OCR
            assert observed.status == STATUS_PROCESSING
            assert observed.progress == progress_for(STAGE_OCR)

    def test_substeps_advance_the_bar_within_a_stage(self, tracker):
        tracker.stage(STAGE_OCR)
        before = tracker.run.progress
        tracker.substep(5, 10)
        assert tracker.run.progress > before
        assert tracker.run.current_stage == STAGE_OCR

    def test_a_substep_outside_a_stage_is_ignored_rather_than_fatal(self, tracker):
        tracker.substep(1, 2)  # must not raise
        assert tracker.run.progress == 0

    def test_stage_timings_accumulate_per_stage(self, tracker):
        tracker.stage(STAGE_VALIDATION)
        tracker.stage(STAGE_OCR)
        tracker.finish(STATUS_COMPLETED)
        assert STAGE_VALIDATION in tracker.run.stage_timings
        assert STAGE_OCR in tracker.run.stage_timings

    def test_a_stage_scope_records_its_timing_even_when_it_raises(self, tracker):
        with pytest.raises(RuntimeError):
            with tracker.stage_scope(STAGE_OCR):
                raise RuntimeError("boom")
        assert STAGE_OCR in tracker.run.stage_timings

    def test_finishing_lands_on_a_hundred_and_a_terminal_status(self, tracker):
        tracker.stage(STAGE_OCR)
        tracker.finish(STATUS_COMPLETED, page_count=3, used_ocr=True, extraction_confidence=0.91)
        assert tracker.run.progress == 100
        assert tracker.run.current_stage == STAGE_DONE
        assert tracker.run.page_count == 3
        assert tracker.run.used_ocr is True
        assert tracker.run.finished_at is not None

    def test_finishing_on_a_non_terminal_status_is_a_programming_error(self, tracker):
        with pytest.raises(ValueError):
            tracker.finish(STATUS_PROCESSING)

    def test_a_failure_is_recorded_with_a_code_a_client_can_branch_on(self, tracker):
        tracker.stage(STAGE_OCR)
        tracker.fail("OCR_UNAVAILABLE", "PaddleOCR could not be initialised")
        assert tracker.run.status == STATUS_FAILED
        assert tracker.run.error_code == "OCR_UNAVAILABLE"
        assert tracker.run.current_stage == STAGE_OCR, "the failing stage must be preserved"
        assert tracker.run.finished_at is not None

    def test_a_failure_survives_a_broken_session(self, tracker, db):
        """The exception that failed the run may have poisoned the session.
        If recording the failure then fails too, the run is stuck in
        PROCESSING forever — the one state a poller cannot recover from."""
        from app.db.session import SessionLocal
        from app.models import Document
        from app.modules.documents.models import DocumentProcessingRun

        db.add(Document(filename="x", object_key="x", content_type="x", document_type="x", owner_id="nonexistent-user"))
        tracker.fail("PIPELINE_ERROR", "something went wrong")

        with SessionLocal() as observer:
            assert observer.get(DocumentProcessingRun, tracker.run.id).status == STATUS_FAILED
