"""Adaptive preprocessing.

What matters here is not that each transform works in isolation — OpenCV's
are fine — but that the *gating* is right. A pipeline that denoises a clean
render or binarises an evenly-lit scan is worse than one that does nothing,
and that failure is invisible without tests that assert on what was skipped.
"""
import cv2
import numpy as np
import pytest

from app.modules.ocr import image_preprocessing_service as preprocessing
from app.modules.ocr.image_preprocessing_service import preprocess


def blank_page(width=1700, height=2200, value=255):
    return np.full((height, width, 3), value, dtype=np.uint8)


def text_page(width=1700, height=2200, lines=30, contrast=0, noise=0.0, blur=0):
    """A synthetic invoice-ish page: dark text lines on a light ground."""
    image = blank_page(width, height, value=255 - contrast)
    ink = 20 + contrast
    for index in range(lines):
        y = 120 + index * 60
        if y + 24 >= height:
            break
        # Varying line lengths, like real text, so projection profiles and
        # contour angles behave the way they do on a real page.
        length = int(width * (0.35 + 0.5 * ((index * 37) % 10) / 10))
        cv2.rectangle(image, (120, y), (120 + length, y + 22), (ink, ink, ink), -1)
    if noise:
        noise_field = np.random.default_rng(1234).normal(0, noise, image.shape)
        image = np.clip(image.astype(np.float64) + noise_field, 0, 255).astype(np.uint8)
    if blur:
        image = cv2.GaussianBlur(image, (blur | 1, blur | 1), 0)
    return image


class TestMeasurement:
    def test_a_clean_page_measures_as_sharp_low_noise_and_high_contrast(self):
        measurements = preprocessing.measure(text_page())

        assert measurements.noise_sigma < preprocessing.NOISE_SIGMA_THRESHOLD
        assert measurements.contrast_spread > preprocessing.LOW_CONTRAST_SPREAD

    def test_added_noise_raises_the_noise_estimate(self):
        clean = preprocessing.measure(text_page()).noise_sigma
        noisy = preprocessing.measure(text_page(noise=14.0)).noise_sigma

        assert noisy > clean
        assert noisy > preprocessing.NOISE_SIGMA_THRESHOLD

    def test_a_washed_out_page_measures_as_low_contrast(self):
        # Text at 150 on a ground at 175: legible to a human, poor for OCR.
        washed = np.full((1200, 900, 3), 175, dtype=np.uint8)
        for index in range(15):
            cv2.rectangle(washed, (80, 80 + index * 60), (700, 100 + index * 60), (150, 150, 150), -1)

        measurements = preprocessing.measure(washed)

        assert measurements.contrast_spread < preprocessing.LOW_CONTRAST_SPREAD

    def test_blur_lowers_the_sharpness_estimate(self):
        assert preprocessing.estimate_sharpness(cv2.cvtColor(text_page(blur=9), cv2.COLOR_BGR2GRAY)) < \
               preprocessing.estimate_sharpness(cv2.cvtColor(text_page(), cv2.COLOR_BGR2GRAY))


class TestSkewEstimation:
    @pytest.mark.parametrize("angle", [-6.0, -2.5, 2.5, 6.0])
    def test_a_known_rotation_is_recovered_with_the_right_sign(self, angle):
        rotated = preprocessing.rotate(text_page(), -angle)

        estimated = preprocessing.estimate_skew(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY))

        assert estimated == pytest.approx(angle, abs=1.0)

    def test_an_upright_page_measures_as_unskewed(self):
        estimated = preprocessing.estimate_skew(cv2.cvtColor(text_page(), cv2.COLOR_BGR2GRAY))
        assert abs(estimated) < preprocessing.MIN_DESKEW_DEGREES

    def test_a_blank_page_reports_no_skew_rather_than_a_random_angle(self):
        # Too few text-line blobs to measure: guessing here would rotate a
        # perfectly good page on the strength of one JPEG artefact.
        assert preprocessing.estimate_skew(cv2.cvtColor(blank_page(), cv2.COLOR_BGR2GRAY)) == 0.0


class TestOrientation:
    def test_a_sideways_page_is_detected(self):
        sideways = np.rot90(text_page(), k=1).copy()

        assert preprocessing.estimate_quarter_turns(cv2.cvtColor(sideways, cv2.COLOR_BGR2GRAY)) == 1

    def test_an_upright_page_is_left_alone(self):
        assert preprocessing.estimate_quarter_turns(cv2.cvtColor(text_page(), cv2.COLOR_BGR2GRAY)) == 0

    def test_a_page_with_no_clear_signal_is_not_rotated(self):
        # A wrong 90-degree turn is far more damaging than leaving a page
        # alone, so ambiguity must resolve to "do nothing".
        assert preprocessing.estimate_quarter_turns(cv2.cvtColor(blank_page(), cv2.COLOR_BGR2GRAY)) == 0

    def test_a_sideways_page_comes_back_upright(self):
        result = preprocess(np.rot90(text_page(), k=1).copy(), allow_perspective=False)

        assert "orientation" in result.applied
        assert result.image.shape[0] > result.image.shape[1], "a portrait page should end up portrait"


class TestUpsideDownResolution:
    def test_the_better_reading_wins(self):
        """Projection profiles cannot separate a page from its 180-degree
        rotation, so only recognition can. Here recognition is stubbed."""
        image = text_page()
        upside_down = np.rot90(image, k=2)

        def score(candidate):
            return 0.9 if np.array_equal(candidate, upside_down) else 0.5

        chosen, flipped = preprocessing.resolve_upside_down(image, score=score)

        assert flipped is True
        assert np.array_equal(chosen, upside_down)

    def test_an_ambiguous_score_leaves_the_page_alone(self):
        # Flipping a correct page is the worse error, so a marginal
        # improvement is not enough to act on.
        image = text_page()

        chosen, flipped = preprocessing.resolve_upside_down(image, score=lambda candidate: 0.80)

        assert flipped is False
        assert chosen is image


class TestGating:
    def test_a_clean_render_is_left_essentially_alone(self):
        """The expensive mistake: denoising and equalising a page that was
        already perfect costs accuracy on the smallest digits."""
        result = preprocess(text_page(), allow_perspective=False)

        assert "denoise" in result.skipped
        assert "contrast" in result.skipped
        assert "deskew" in result.skipped
        assert "binarize" in result.skipped

    def test_a_noisy_page_is_denoised(self):
        result = preprocess(text_page(noise=14.0), allow_perspective=False)

        assert "denoise" in result.applied

    def test_a_soft_page_is_not_denoised_however_noisy_it_measures(self):
        # Denoising an already-soft image destroys the little edge signal
        # left, so sharpness gates the noise decision.
        result = preprocess(text_page(noise=14.0, blur=11), allow_perspective=False)

        assert "denoise" in result.skipped

    def test_a_washed_out_page_gets_contrast_enhancement(self):
        washed = np.full((1800, 1300, 3), 175, dtype=np.uint8)
        for index in range(20):
            cv2.rectangle(washed, (100, 100 + index * 70), (900, 125 + index * 70), (150, 150, 150), -1)

        result = preprocess(washed, allow_perspective=False)

        assert "contrast" in result.applied

    def test_a_skewed_page_is_straightened(self):
        result = preprocess(preprocessing.rotate(text_page(), -3.0), allow_perspective=False)

        assert "deskew" in result.applied
        residual = preprocessing.estimate_skew(cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY))
        assert abs(residual) < 1.0, "the page should come out very nearly straight"

    def test_a_barely_skewed_page_is_not_rotated(self):
        # Rotating costs an interpolation pass; below a third of a degree
        # there is nothing to gain and detail to lose.
        result = preprocess(preprocessing.rotate(text_page(), -0.1), allow_perspective=False)

        assert "deskew" in result.skipped

    def test_a_low_resolution_scan_is_upscaled(self):
        result = preprocess(text_page(width=700, height=900, lines=10), allow_perspective=False)

        assert "upscale" in result.applied
        assert min(result.image.shape[:2]) >= 900

    def test_an_already_large_page_is_not_upscaled(self):
        result = preprocess(text_page(width=2480, height=3508), allow_perspective=False)

        assert "upscale" in result.skipped

    def test_upscaling_is_capped_so_a_thumbnail_is_not_blown_up_absurdly(self):
        tiny = text_page(width=200, height=280, lines=3)

        upscaled = preprocessing.upscale(tiny)

        assert upscaled.shape[1] <= tiny.shape[1] * preprocessing.MAX_UPSCALE_FACTOR + 1

    def test_binarisation_is_off_unless_asked_for(self):
        washed = np.full((1400, 1000, 3), 140, dtype=np.uint8)
        for index in range(15):
            cv2.rectangle(washed, (80, 80 + index * 70), (800, 105 + index * 70), (128, 128, 128), -1)

        assert "binarize" in preprocess(washed, allow_perspective=False).skipped


class TestPerspective:
    def test_a_rendered_page_is_never_warped(self):
        """A PDF render has no page boundary in the frame, so any
        quadrilateral found is a table border — warping to it is destructive."""
        result = preprocess(text_page(), allow_perspective=False)

        assert "perspective" in result.skipped

    def test_a_straight_on_photograph_is_not_warped(self):
        # The page fills the frame as a rectangle already; rectifying it
        # costs a resample and returns the same pixels, blurrier.
        photo = np.full((1600, 1200, 3), 60, dtype=np.uint8)
        photo[60:1540, 60:1140] = text_page(width=1080, height=1480, lines=18)

        result = preprocess(photo)

        assert "perspective" in result.skipped

    def test_corner_ordering_is_stable_regardless_of_input_order(self):
        corners = np.array([[300, 20], [20, 40], [320, 400], [10, 380]], dtype=np.float32)

        ordered = preprocessing._order_corners(corners)

        assert ordered[0].tolist() == [20, 40]     # top-left
        assert ordered[1].tolist() == [300, 20]    # top-right
        assert ordered[2].tolist() == [320, 400]   # bottom-right
        assert ordered[3].tolist() == [10, 380]    # bottom-left


class TestContract:
    def test_the_original_image_is_never_mutated(self):
        """The original is what gets stored, shown to a reviewer and
        reprocessed later; overwriting it makes past extractions
        unreproducible."""
        original = text_page(noise=14.0)
        untouched = original.copy()

        preprocess(original, allow_perspective=False)

        assert np.array_equal(original, untouched)

    def test_the_result_reports_both_what_ran_and_what_did_not(self):
        result = preprocess(text_page(), allow_perspective=False)

        every_step = set(result.applied) | set(result.skipped)
        assert {"orientation", "perspective", "deskew", "upscale", "denoise", "contrast", "binarize"} <= every_step
        assert not set(result.applied) & set(result.skipped), "a step cannot be both applied and skipped"

    def test_measurements_are_carried_for_diagnosis(self):
        result = preprocess(text_page(), allow_perspective=False)

        payload = result.to_dict()
        assert payload["measurements"]["contrast_spread"] > 0
        assert payload["measurements"]["width"] > 0

    def test_total_rotation_is_reported_so_boxes_can_be_mapped_back(self):
        result = preprocess(preprocessing.rotate(text_page(), -3.0), allow_perspective=False)

        assert result.rotation_applied == pytest.approx(3.0, abs=1.0)

    def test_the_output_stays_a_three_channel_image_for_the_recogniser(self):
        result = preprocess(text_page(noise=14.0), allow_perspective=False)

        assert result.image.ndim == 3 and result.image.shape[2] == 3

    def test_an_empty_image_is_rejected_rather_than_producing_garbage(self):
        with pytest.raises(ValueError):
            preprocess(np.array([]))
