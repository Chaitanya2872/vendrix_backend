"""CPU-safe document and vehicle image recognition helpers.

OCR itself now lives in `app.modules.ocr` — this module keeps the small
image utilities the vehicle/ANPR path uses and re-exports `read_text` so the
existing callers (ANPR, the vendor-document worker, the invoice text
extractor and the ML corpus builder) keep working unchanged while new code
consumes the structured `OcrPage` from the OCR service instead of a flat
string.

The oneDNN workaround that used to live here moved with the engine: it must
run before paddle is imported, and `app.modules.ocr.engine` is now the only
place that imports paddle at all.
"""
import re
import cv2
import numpy as np

from app.modules.ocr import service as ocr_service


def decode_image(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None: raise ValueError("Unable to decode image")
    return image


def enhance(image: np.ndarray) -> np.ndarray:
    """Fixed denoise + contrast pass, retained for the ANPR path.

    Invoice pages go through `image_preprocessing_service` instead, which
    measures the page before deciding what to apply — a number plate crop and
    a scanned A4 invoice do not want the same treatment, and applying this
    unconditionally to clean scans costs accuracy.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)


def read_text(image: np.ndarray) -> str:
    """Flat reading-order text for callers that predate structured OCR.

    New code should call `app.modules.ocr.service.recognize_page` and use the
    boxes and confidences; this wrapper exists so that switching the engine
    did not require changing every existing caller at once.
    """
    return ocr_service.recognize_page(enhance(image)).text


def registration_candidates(text: str) -> list[str]:
    values = re.findall(r"(?:[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4})", text.upper())
    return [re.sub(r"[^A-Z0-9]", "", value) for value in values]


def crop_plate(image: np.ndarray, model_path: str | None) -> np.ndarray:
    if not model_path: return image
    from ultralytics import YOLO
    result = YOLO(model_path)(image, verbose=False)[0]
    if not result.boxes or len(result.boxes) == 0: return image
    x1, y1, x2, y2 = map(int, result.boxes.xyxy[0].tolist())
    return image[max(0,y1):y2, max(0,x1):x2]
