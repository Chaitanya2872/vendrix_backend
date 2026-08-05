"""CPU-safe document and vehicle image recognition helpers."""
from functools import lru_cache
import re
import cv2
import numpy as np


def decode_image(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None: raise ValueError("Unable to decode image")
    return image


def enhance(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


@lru_cache
def _ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)


def read_text(image: np.ndarray) -> str:
    result = _ocr().predict(enhance(image))
    lines: list[str] = []
    for page in result:
        data = page.json if hasattr(page, "json") else page
        payload = data.get("res", data) if isinstance(data, dict) else {}
        lines.extend(payload.get("rec_texts", []))
    return "\n".join(lines)


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
