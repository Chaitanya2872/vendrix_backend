"""Engine adapter: tolerating PaddleOCR's result shapes.

A paddle upgrade that changes the result shape must not turn into silently
empty extractions — an invoice pipeline that returns "no text found" for
every document looks like bad scans, not a broken adapter. These tests pin
both shapes the adapter claims to accept.
"""
import numpy as np
import pytest

from app.modules.ocr import engine
from app.modules.ocr.exceptions import OcrPageFailed


class ResultV3(dict):
    """Stand-in for a PaddleOCR 3.x result: dict-like, and also exposes
    `.json` the way paddle's own result objects do."""

    @property
    def json(self):
        return {"res": dict(self)}


class ResultJsonOnly:
    """A result object that is not dict-like — only reachable via `.json`."""

    def __init__(self, payload):
        self._payload = payload

    @property
    def json(self):
        return {"res": self._payload}


def square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class TestVersion3Shape:
    def test_reads_texts_scores_and_polygons(self):
        result = [ResultV3({
            "rec_texts": ["Tax Invoice", "INV-42"],
            "rec_scores": [0.99, 0.87],
            "rec_polys": [square(10, 10, 200, 40), square(10, 60, 120, 85)],
        })]
        detections = engine._normalise_result(result)
        assert [d.text for d in detections] == ["Tax Invoice", "INV-42"]
        assert [d.confidence for d in detections] == [0.99, 0.87]
        assert detections[0].polygon == [(10.0, 10.0), (200.0, 10.0), (200.0, 40.0), (10.0, 40.0)]

    def test_numpy_polygons_are_converted_to_plain_floats(self):
        # numpy scalars leak into JSON encoders and equality checks, and only
        # blow up at the API boundary three stages later.
        result = [ResultV3({
            "rec_texts": ["INV-42"],
            "rec_scores": [np.float32(0.87)],
            "rec_polys": [np.array(square(10, 60, 120, 85), dtype=np.float32)],
        })]
        detection = engine._normalise_result(result)[0]
        assert detection.polygon == [(10.0, 60.0), (120.0, 60.0), (120.0, 85.0), (10.0, 85.0)]
        assert all(type(value) is float for point in detection.polygon for value in point)
        assert type(detection.confidence) is float

    def test_falls_back_to_detection_polygons_when_recognition_polygons_are_absent(self):
        result = [ResultV3({
            "rec_texts": ["INV-42"],
            "rec_scores": [0.9],
            "rec_polys": [],
            "dt_polys": [square(10, 60, 120, 85)],
        })]
        assert len(engine._normalise_result(result)) == 1

    def test_a_detection_without_geometry_is_dropped_not_guessed(self):
        # Unusable downstream: a box is how every later stage locates it.
        result = [ResultV3({
            "rec_texts": ["has geometry", "no geometry"],
            "rec_scores": [0.9, 0.9],
            "rec_polys": [square(10, 10, 100, 30)],
        })]
        assert [d.text for d in engine._normalise_result(result)] == ["has geometry"]

    def test_missing_scores_default_to_zero_rather_than_faking_certainty(self):
        result = [ResultV3({"rec_texts": ["INV-42"], "rec_polys": [square(0, 0, 10, 10)]})]
        assert engine._normalise_result(result)[0].confidence == 0.0

    def test_reads_a_result_reachable_only_through_json(self):
        result = [ResultJsonOnly({
            "rec_texts": ["Tax Invoice"],
            "rec_scores": [0.95],
            "rec_polys": [square(10, 10, 200, 40)],
        })]
        assert [d.text for d in engine._normalise_result(result)] == ["Tax Invoice"]

    def test_multiple_pages_in_one_result_are_flattened(self):
        result = [
            ResultV3({"rec_texts": ["page one"], "rec_scores": [0.9], "rec_polys": [square(0, 0, 10, 10)]}),
            ResultV3({"rec_texts": ["page two"], "rec_scores": [0.9], "rec_polys": [square(0, 0, 10, 10)]}),
        ]
        assert [d.text for d in engine._normalise_result(result)] == ["page one", "page two"]


class TestVersion2Shape:
    def test_reads_the_legacy_nested_list_format(self):
        result = [[
            [square(10, 10, 200, 40), ("Tax Invoice", 0.99)],
            [square(10, 60, 120, 85), ("INV-42", 0.87)],
        ]]
        detections = engine._normalise_result(result)
        assert [d.text for d in detections] == ["Tax Invoice", "INV-42"]
        assert [d.confidence for d in detections] == [0.99, 0.87]

    def test_malformed_legacy_entries_are_skipped_not_fatal(self):
        result = [[
            [square(10, 10, 200, 40), ("good", 0.99)],
            ["not a polygon"],
        ]]
        assert [d.text for d in engine._normalise_result(result)] == ["good"]


class TestUnknownShape:
    def test_an_unrecognised_shape_yields_nothing_rather_than_raising(self):
        # Logged loudly, but one odd page must not abort a document.
        assert engine._normalise_result([object()]) == []

    def test_an_empty_result_is_not_an_error(self):
        assert engine._normalise_result([]) == []


class TestGuards:
    def test_an_empty_image_is_rejected_before_the_engine_is_loaded(self):
        # Loading paddle costs seconds and hundreds of megabytes; a caller
        # that passes an empty array should not pay for it.
        with pytest.raises(OcrPageFailed):
            engine.recognize(np.array([]))

    def test_a_none_image_is_rejected(self):
        with pytest.raises(OcrPageFailed):
            engine.recognize(None)


class TestOneDnnWorkaround:
    def test_the_crash_workaround_is_set_at_import_time(self):
        # Must precede any paddle import in the process, which is the entire
        # reason this module is the only one that imports paddle.
        import os
        assert os.environ.get("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT") == "0"
