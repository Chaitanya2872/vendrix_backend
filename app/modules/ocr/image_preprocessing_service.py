"""Adaptive preprocessing: measure the page, then apply only what it needs.

The naive pipeline applies denoise + CLAHE + threshold to everything, and it
loses accuracy on exactly the documents that were fine to begin with —
denoising a clean 300-DPI render smears the thin strokes of 8pt digits, and
Otsu on an evenly-lit scan turns light grey text into nothing. Every
transform here is therefore gated on a measurement, and the measurements are
returned alongside the image so a bad decision can be diagnosed rather than
guessed at.

Order is deliberate: page-level geometry first (orientation, perspective,
skew), then resolution, then appearance. Correcting skew after upscaling
costs four times the interpolation for the same answer, and measuring
contrast before fixing perspective measures the background as much as the
page.

Nothing here mutates its input. The original is what gets stored, shown to a
reviewer, and re-processed after a model upgrade; a pipeline that overwrote
it would make every past extraction unreproducible.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --- thresholds, all derived from what the transform is protecting against ---

# Below this short-edge pixel count, 8pt invoice text has too few pixels per
# stroke for the recogniser. 1600px on the short edge ≈ 200 DPI on A4.
MIN_SHORT_EDGE = 1600
# Never upscale beyond this: interpolation invents no detail, and OCR runtime
# grows with pixel count.
MAX_SHORT_EDGE = 3500
MAX_UPSCALE_FACTOR = 3.0

# Skew below this is not worth the interpolation loss of rotating.
MIN_DESKEW_DEGREES = 0.3
# Above this, the estimate is more likely a bad measurement than a real skew
# — a genuinely 20-degree page is a photo, and perspective correction owns it.
MAX_DESKEW_DEGREES = 15.0

# Laplacian variance below this means the image is soft/blurred; denoising a
# soft image destroys what little edge signal is left.
MIN_SHARPNESS_FOR_DENOISE = 60.0
# Estimated noise sigma above this earns a denoise pass.
NOISE_SIGMA_THRESHOLD = 3.5

# Contrast, as the spread between the 5th and 95th intensity percentiles.
# A well-exposed scan sits near 200; below this the page is washed out.
LOW_CONTRAST_SPREAD = 110

# A page contour must cover at least this fraction of the frame before it is
# treated as a photographed document worth rectifying.
MIN_PAGE_CONTOUR_AREA = 0.40
# ...and its corners must deviate from a rectangle by at least this fraction
# of the frame width, or the "correction" is a no-op that costs a resample.
MIN_PERSPECTIVE_SKEW = 0.02


@dataclass
class PageMeasurements:
    """What was measured, and therefore why each decision was made."""

    width: int
    height: int
    short_edge: int
    is_colour: bool
    sharpness: float = 0.0
    noise_sigma: float = 0.0
    contrast_spread: float = 0.0
    mean_intensity: float = 0.0
    skew_degrees: float = 0.0
    quarter_turns: int = 0
    page_contour_area: float = 0.0

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "short_edge": self.short_edge,
            "is_colour": self.is_colour,
            "sharpness": round(self.sharpness, 2),
            "noise_sigma": round(self.noise_sigma, 3),
            "contrast_spread": round(self.contrast_spread, 1),
            "mean_intensity": round(self.mean_intensity, 1),
            "skew_degrees": round(self.skew_degrees, 3),
            "quarter_turns": self.quarter_turns,
            "page_contour_area": round(self.page_contour_area, 3),
        }


@dataclass
class PreprocessedPage:
    image: np.ndarray
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    measurements: PageMeasurements | None = None
    rotation_applied: float = 0.0

    def to_dict(self) -> dict:
        return {
            "applied": list(self.applied),
            "skipped": list(self.skipped),
            "rotation_applied": round(self.rotation_applied, 3),
            "measurements": self.measurements.to_dict() if self.measurements else None,
        }


# --- measurement -----------------------------------------------------------


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard blur proxy. High on crisp
    text, low on a soft photo or an over-denoised scan."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkær's noise estimate: convolve with a mask orthogonal to the
    Laplacian, so structure cancels and noise does not.

    Preferred over "variance of a flat patch" because an invoice has no
    reliably flat patch — the margins carry scanner gradients and the body is
    all edges.
    """
    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return 0.0
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    convolved = cv2.filter2D(gray.astype(np.float64), -1, mask)
    sigma = np.sum(np.abs(convolved))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6.0 * (width - 2) * (height - 2))
    return float(sigma)


def estimate_contrast_spread(gray: np.ndarray) -> float:
    """Intensity spread between the 5th and 95th percentiles.

    Percentiles rather than min/max: a single dust speck at 0 and a
    specular highlight at 255 would otherwise report a perfectly exposed
    page and a washed-out one identically.
    """
    low, high = np.percentile(gray, [5, 95])
    return float(high - low)


def estimate_skew(gray: np.ndarray) -> float:
    """Skew angle in degrees, positive meaning the page leans clockwise.

    Text lines are found by binarising and dilating horizontally so words
    merge into line-shaped blobs, then taking the *median* angle of those
    blobs' minimum-area rectangles. Median rather than mean because a logo,
    a stamp or a vertical rule contributes a wildly wrong angle, and one such
    outlier drags a mean far enough to make the correction worse than none.
    """
    height, width = gray.shape[:2]
    if height < 50 or width < 50:
        return 0.0

    # Work at a reduced size: skew is a page-level property and the estimate
    # is unchanged by detail, while the morphology cost is quadratic.
    scale = min(1.0, 1000.0 / max(height, width))
    working = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray

    binary = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles: list[float] = []
    for contour in contours:
        (_, _), (rect_width, rect_height), angle = cv2.minAreaRect(contour)
        # Filter on long/short side, not width/height: OpenCV swaps which
        # dimension it calls the width depending on tilt direction, so a
        # `rect_width < 40` test discards every line tilted one particular
        # way and reports that page as perfectly straight.
        long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
        if long_side < 40 or short_side < 3:
            continue  # too small to be a text line
        # minAreaRect's angle convention flips with which side it calls the
        # width, so a line tilted one way comes back as ~6 degrees and the
        # same tilt the other way as ~84. Normalise to "how far is the long
        # axis from horizontal", wrapped into (-45, 45]. Wrapping by
        # arithmetic rather than a single conditional subtraction: a lone
        # `if angle > 45: angle -= 90` leaves the 84-degree case at 84, where
        # the sanity filter below silently discards it — which drops every
        # counter-clockwise line and reports a skewed page as straight.
        if rect_width < rect_height:
            angle -= 90
        angle = ((angle + 45) % 90) - 45
        if abs(angle) <= MAX_DESKEW_DEGREES:
            angles.append(angle)

    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def estimate_quarter_turns(gray: np.ndarray) -> int:
    """How many 90-degree turns the page needs, or 0 if it is upright.

    Text lines make the projection profile perpendicular to them strongly
    periodic — bands of ink separated by bands of paper — while the profile
    parallel to them is comparatively flat. Comparing the two variances tells
    upright from sideways without any extra model or system binary.

    This deliberately cannot tell upright from upside-down: both have
    horizontal lines and identical profiles. Resolving 180 degrees needs
    recognition, so it is left to `resolve_upside_down`, which the OCR stage
    can call when it has a confidence score to compare.
    """
    height, width = gray.shape[:2]
    if height < 50 or width < 50:
        return 0

    scale = min(1.0, 800.0 / max(height, width))
    working = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    binary = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal = binary.sum(axis=1).astype(np.float64)   # ink per row
    vertical = binary.sum(axis=0).astype(np.float64)     # ink per column

    def periodicity(profile: np.ndarray) -> float:
        if profile.size < 8 or profile.max() <= 0:
            return 0.0
        normalised = profile / profile.max()
        # Mean absolute successive difference: high when the profile
        # alternates ink/paper rapidly, near zero when it is smooth.
        return float(np.mean(np.abs(np.diff(normalised))))

    row_signal = periodicity(horizontal)
    column_signal = periodicity(vertical)

    # A clear margin is required before rotating: on a sparse page the two
    # signals are close, and a wrong 90-degree turn is far more damaging than
    # leaving a page alone.
    if row_signal > column_signal * 1.35:
        return 0      # lines run horizontally: already upright
    if column_signal > row_signal * 1.35:
        return 1      # lines run vertically: page is on its side
    return 0


def find_page_quadrilateral(image: np.ndarray) -> np.ndarray | None:
    """Locate a photographed page's four corners, or None.

    Returns None for scans and renders — they have no page boundary inside
    the frame, and warping one is a pure loss.
    """
    gray = _to_gray(image)
    height, width = gray.shape[:2]
    frame_area = float(height * width)
    if frame_area <= 0:
        return None

    scale = min(1.0, 900.0 / max(height, width))
    working = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray

    blurred = cv2.GaussianBlur(working, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    working_area = float(working.shape[0] * working.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if area / working_area < MIN_PAGE_CONTOUR_AREA:
            continue
        approximated = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approximated) != 4 or not cv2.isContourConvex(approximated):
            continue
        corners = approximated.reshape(4, 2).astype(np.float32)
        return corners / scale if scale < 1.0 else corners
    return None


def measure(image: np.ndarray) -> PageMeasurements:
    """Everything the decisions below depend on, computed once."""
    gray = _to_gray(image)
    height, width = gray.shape[:2]
    quadrilateral = find_page_quadrilateral(image)
    return PageMeasurements(
        width=int(width),
        height=int(height),
        short_edge=int(min(width, height)),
        is_colour=image.ndim == 3 and image.shape[2] >= 3,
        sharpness=estimate_sharpness(gray),
        noise_sigma=estimate_noise_sigma(gray),
        contrast_spread=estimate_contrast_spread(gray),
        mean_intensity=float(gray.mean()),
        skew_degrees=estimate_skew(gray),
        quarter_turns=estimate_quarter_turns(gray),
        page_contour_area=(
            float(cv2.contourArea(quadrilateral.astype(np.float32))) / float(height * width)
            if quadrilateral is not None else 0.0
        ),
    )


# --- transforms ------------------------------------------------------------


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Sort four corners into top-left, top-right, bottom-right, bottom-left.

    By coordinate sums and differences rather than by angle: it is exact for
    any convex quadrilateral and needs no centroid, which is unstable when
    one corner is far outside the others.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = corners.sum(axis=1)
    difference = np.diff(corners, axis=1).ravel()
    ordered[0] = corners[np.argmin(total)]       # top-left: smallest x+y
    ordered[2] = corners[np.argmax(total)]       # bottom-right: largest x+y
    ordered[1] = corners[np.argmin(difference)]  # top-right: smallest y-x
    ordered[3] = corners[np.argmax(difference)]  # bottom-left: largest y-x
    return ordered


def correct_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    ordered = _order_corners(corners)
    top_left, top_right, bottom_right, bottom_left = ordered

    width = int(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    height = int(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
    if width < 10 or height < 10:
        return image

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the centre, expanding the canvas so no text is clipped.

    The border is filled by replication rather than black: a black wedge in
    the corner of an otherwise white page is a strong edge that the text
    detector will happily propose as a text region.
    """
    height, width = image.shape[:2]
    centre = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(centre, degrees, 1.0)

    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - centre[0]
    matrix[1, 2] += new_height / 2 - centre[1]

    return cv2.warpAffine(
        image, matrix, (new_width, new_height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def upscale(image: np.ndarray, target_short_edge: int = MIN_SHORT_EDGE) -> np.ndarray:
    height, width = image.shape[:2]
    short_edge = min(height, width)
    if short_edge <= 0:
        return image
    factor = min(target_short_edge / short_edge, MAX_UPSCALE_FACTOR, MAX_SHORT_EDGE / short_edge)
    if factor <= 1.01:
        return image
    # Cubic, not Lanczos: Lanczos rings around high-contrast text edges, and
    # the ringing reads as ink to the detector.
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def denoise(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    # Small h: aggressive non-local-means erases the thin strokes of small
    # digits, which is precisely the text whose accuracy matters most here.
    cleaned = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else cleaned


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE on the luminance channel only.

    Local rather than global equalisation because scanner and phone lighting
    fall off across the page: a global stretch fixes the bright half and
    crushes the dark half.
    """
    gray = _to_gray(image)
    equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(equalised, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else equalised


def binarize(image: np.ndarray) -> np.ndarray:
    """Adaptive (Sauvola-like) binarisation for badly-lit pages.

    Rarely used: PaddleOCR is trained on natural images and generally does
    better on greyscale than on anything binarised, so this is reserved for
    the classic bad-fax case where the text and background intensities
    overlap globally but separate locally.
    """
    gray = _to_gray(image)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=35, C=15
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else binary


# --- the pipeline ----------------------------------------------------------


def preprocess(
    image: np.ndarray,
    allow_perspective: bool = True,
    allow_binarize: bool = False,
) -> PreprocessedPage:
    """Prepare one page image for OCR, applying only what it needs.

    `allow_perspective` is switched off for rendered PDF pages: they have no
    page boundary inside the frame, so any quadrilateral found there is a
    table border or a logo box, and warping to it destroys the page.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot preprocess an empty image")

    working = image.copy()
    applied: list[str] = []
    skipped: list[str] = []
    total_rotation = 0.0

    measurements = measure(working)

    # 1. Orientation — before anything that measures or resamples geometry.
    if measurements.quarter_turns:
        working = np.rot90(working, k=-measurements.quarter_turns).copy()
        total_rotation += -90.0 * measurements.quarter_turns
        applied.append("orientation")
    else:
        skipped.append("orientation")

    # 2. Perspective — photographs only.
    if allow_perspective:
        corners = find_page_quadrilateral(working)
        if corners is not None and _is_meaningfully_skewed(corners, working.shape):
            working = correct_perspective(working, corners)
            applied.append("perspective")
        else:
            skipped.append("perspective")
    else:
        skipped.append("perspective")

    # 3. Deskew — re-measured, because the two steps above moved the page.
    skew = estimate_skew(_to_gray(working))
    if MIN_DESKEW_DEGREES <= abs(skew) <= MAX_DESKEW_DEGREES:
        working = rotate(working, skew)
        total_rotation += skew
        applied.append("deskew")
    else:
        skipped.append("deskew")

    # 4. Resolution.
    if min(working.shape[:2]) < MIN_SHORT_EDGE:
        before = working.shape[:2]
        working = upscale(working)
        if working.shape[:2] != before:
            applied.append("upscale")
        else:
            skipped.append("upscale")
    else:
        skipped.append("upscale")

    # 5. Noise — only on an image sharp enough to survive it.
    if measurements.noise_sigma > NOISE_SIGMA_THRESHOLD and measurements.sharpness >= MIN_SHARPNESS_FOR_DENOISE:
        working = denoise(working)
        applied.append("denoise")
    else:
        skipped.append("denoise")

    # 6. Contrast.
    if measurements.contrast_spread < LOW_CONTRAST_SPREAD:
        working = enhance_contrast(working)
        applied.append("contrast")
    else:
        skipped.append("contrast")

    # 7. Binarisation — opt-in only.
    if allow_binarize and measurements.contrast_spread < LOW_CONTRAST_SPREAD / 2:
        working = binarize(working)
        applied.append("binarize")
    else:
        skipped.append("binarize")

    logger.info(
        "preprocess.completed applied=%s skipped=%s skew=%.2f noise=%.2f contrast=%.0f",
        applied, skipped, measurements.skew_degrees, measurements.noise_sigma,
        measurements.contrast_spread,
    )
    return PreprocessedPage(
        image=working,
        applied=applied,
        skipped=skipped,
        measurements=measurements,
        rotation_applied=total_rotation,
    )


def _is_meaningfully_skewed(corners: np.ndarray, shape: tuple) -> bool:
    """Whether rectifying this quadrilateral would change anything.

    A page photographed straight-on yields a quadrilateral that is already a
    rectangle; warping it costs a full resample and returns the same pixels
    slightly blurrier.
    """
    ordered = _order_corners(corners)
    frame_width = float(shape[1])
    if frame_width <= 0:
        return False
    top_left, top_right, bottom_right, bottom_left = ordered
    deviations = [
        abs(top_left[1] - top_right[1]),
        abs(bottom_left[1] - bottom_right[1]),
        abs(top_left[0] - bottom_left[0]),
        abs(top_right[0] - bottom_right[0]),
    ]
    return max(deviations) / frame_width >= MIN_PERSPECTIVE_SKEW


def resolve_upside_down(
    image: np.ndarray,
    score: "callable",
) -> tuple[np.ndarray, bool]:
    """Decide between a page and the same page rotated 180 degrees.

    Projection profiles cannot tell these apart — both have horizontal text
    lines — so the only honest test is to recognise both and keep the better
    reading. `score` is injected rather than imported so this module stays
    free of any OCR dependency, and so the caller decides how expensive a
    scoring pass to pay for (a downscaled page is usually enough).

    Returns the chosen image and whether it was flipped.
    """
    upright_score = score(image)
    flipped = np.rot90(image, k=2).copy()
    flipped_score = score(flipped)

    # A clear margin is required: on a page with little text the two scores
    # are noise, and flipping a correct page is the worse error.
    if flipped_score > upright_score * 1.15:
        logger.info("preprocess.flipped_180 upright=%.3f flipped=%.3f", upright_score, flipped_score)
        return flipped, True
    return image, False
