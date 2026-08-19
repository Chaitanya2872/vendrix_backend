"""Scan/photo degradations applied to rendered invoice pages.

A model trained only on crisp 300-dpi renders learns to read crisp 300-dpi
renders. The uploads this pipeline actually receives are phone photos and
office-scanner output: skewed a degree or two, unevenly lit, blurred, noisy
and re-compressed. Those artefacts are exactly what makes OCR drop a decimal
point or turn a 0 into an O, so the training corpus has to contain them —
otherwise the measured accuracy is a number about a situation that never
occurs.

Every transform is seeded, so a degraded page can be reproduced from its
document id and format alone.
"""
from __future__ import annotations

import cv2
import numpy as np

# Severity tiers. Sampling across tiers rather than fixing one lets the
# evaluation report show how accuracy decays as input quality drops, which is
# more useful than a single averaged number.
SEVERITIES: tuple[str, ...] = ("clean", "light", "medium", "heavy")

_PARAMS: dict[str, dict[str, tuple[float, float]]] = {
    # (min, max) for each effect; "clean" still gets a touch of noise because
    # even a good flatbed scan is not a bit-perfect copy of the render.
    "clean":  {"rotate": (-0.3, 0.3), "blur": (0.0, 0.4), "noise": (1.0, 3.0),
               "brightness": (-6, 6), "contrast": (0.98, 1.02), "warp": (0.0, 0.002),
               "jpeg": (88, 95), "scale": (0.95, 1.0), "shadow": (0.0, 0.06)},
    "light":  {"rotate": (-0.9, 0.9), "blur": (0.3, 0.8), "noise": (3.0, 7.0),
               "brightness": (-14, 14), "contrast": (0.94, 1.08), "warp": (0.001, 0.005),
               "jpeg": (72, 88), "scale": (0.82, 0.95), "shadow": (0.05, 0.16)},
    "medium": {"rotate": (-1.8, 1.8), "blur": (0.6, 1.4), "noise": (6.0, 12.0),
               "brightness": (-24, 24), "contrast": (0.88, 1.14), "warp": (0.003, 0.010),
               "jpeg": (55, 75), "scale": (0.68, 0.85), "shadow": (0.12, 0.28)},
    "heavy":  {"rotate": (-3.0, 3.0), "blur": (1.1, 2.1), "noise": (10.0, 20.0),
               "brightness": (-34, 34), "contrast": (0.80, 1.22), "warp": (0.008, 0.018),
               "jpeg": (38, 58), "scale": (0.55, 0.72), "shadow": (0.22, 0.42)},
}


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(rng.uniform(low, high))


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def _perspective(image: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """A page photographed rather than scanned is never perfectly flat to the
    lens; a few pixels of keystone is what that looks like."""
    if amount <= 0:
        return image
    height, width = image.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    jitter = amount * min(width, height)
    target = source + rng.uniform(-jitter, jitter, source.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def _shadow(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Uneven illumination: a soft linear gradient across a random axis. This
    is the single most common failure mode for phone photos of paper, and it
    is what the CLAHE step in vision.enhance() exists to undo."""
    if strength <= 0:
        return image
    height, width = image.shape[:2]
    axis = rng.random()
    ramp_x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    ramp_y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    ramp = axis * ramp_x + (1 - axis) * ramp_y
    if rng.random() < 0.5:
        ramp = 1.0 - ramp
    mask = (1.0 - strength * ramp)[:, :, None]
    return np.clip(image.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def _jpeg_cycle(image: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip through JPEG. Documents are routinely re-saved by scanner
    software and messaging apps before they ever reach an upload form."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def degrade(image: np.ndarray, severity: str, seed: int) -> np.ndarray:
    """Apply the full degradation chain at the given severity.

    Order matters and mirrors physical reality: the page is deformed and lit
    (geometry, shadow), then captured (scale, blur, sensor noise, exposure),
    then compressed. Compressing before blurring would hide the compression
    artefacts the OCR engine actually has to cope with.
    """
    if severity not in _PARAMS:
        raise ValueError(f"Unknown severity '{severity}'; expected one of {SEVERITIES}")
    params = _PARAMS[severity]
    rng = np.random.default_rng(seed)

    result = _perspective(image, _uniform(rng, params["warp"]), rng)
    result = _rotate(result, _uniform(rng, params["rotate"]))
    result = _shadow(result, _uniform(rng, params["shadow"]), rng)

    scale = _uniform(rng, params["scale"])
    if scale < 0.999:
        height, width = result.shape[:2]
        small = cv2.resize(result, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        result = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    sigma = _uniform(rng, params["blur"])
    if sigma > 0.05:
        result = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma, sigmaY=sigma)

    contrast = _uniform(rng, params["contrast"])
    brightness = _uniform(rng, params["brightness"])
    result = np.clip(result.astype(np.float32) * contrast + brightness, 0, 255)

    noise = rng.normal(0.0, _uniform(rng, params["noise"]), result.shape)
    result = np.clip(result + noise, 0, 255).astype(np.uint8)

    return _jpeg_cycle(result, int(_uniform(rng, params["jpeg"])))


def severity_for(seed: int) -> str:
    """Pick a severity for a document. Weighted towards the middle: a corpus
    that is mostly pristine overstates accuracy, one that is mostly illegible
    trains the model on noise."""
    rng = np.random.default_rng(seed)
    return str(rng.choice(SEVERITIES, p=[0.20, 0.35, 0.30, 0.15]))
