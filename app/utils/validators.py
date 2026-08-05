import re
from fastapi import HTTPException


def normalize_registration_number(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", value.upper())
    if not 8 <= len(normalized) <= 11:
        raise HTTPException(422, detail="Invalid registration-number length")
    return normalized
