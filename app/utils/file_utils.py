"""Upload validation: is this file what it claims to be, and can we read it?

The rule here is that **nothing the client sends is evidence**. A browser
derives Content-Type from the file extension, and both are attacker-supplied
in a direct API call. So the ladder is: cheap checks first to reject the
obvious (extension, declared MIME, size), then the only check that actually
establishes the format — the bytes themselves — then an integrity open to
separate "a real PDF" from "the first five bytes of a PDF".

Signature sniffing is implemented here rather than via python-magic because
libmagic is an awkward system dependency on Windows, and this project needs
exactly six formats. A general-purpose type oracle would be more code to
install and no more correct for the cases that matter.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Formats the invoice pipeline can process, and the extension each is
# canonically written with. TIFF is included per the on-premise scanner
# workflow: departmental scanners still default to multi-page TIFF.
PDF = "pdf"
JPEG = "jpeg"
PNG = "png"
TIFF = "tiff"
WEBP = "webp"
DOCX = "docx"
XLSX = "xlsx"

EXTENSIONS: dict[str, set[str]] = {
    PDF: {".pdf"},
    JPEG: {".jpg", ".jpeg"},
    PNG: {".png"},
    TIFF: {".tif", ".tiff"},
    WEBP: {".webp"},
    DOCX: {".docx"},
    XLSX: {".xlsx"},
}

MEDIA_TYPES: dict[str, set[str]] = {
    PDF: {"application/pdf"},
    JPEG: {"image/jpeg", "image/jpg", "image/pjpeg"},
    PNG: {"image/png"},
    TIFF: {"image/tiff", "image/tif", "image/x-tiff"},
    WEBP: {"image/webp"},
    DOCX: {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    XLSX: {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
}

# Formats that carry raster pixels and therefore need OCR.
RASTER_FORMATS = frozenset({JPEG, PNG, TIFF, WEBP})
# Formats whose text is exact and needs no OCR.
EXACT_TEXT_FORMATS = frozenset({DOCX, XLSX})

EXTENSION_TO_FORMAT: dict[str, str] = {
    extension: file_format
    for file_format, extensions in EXTENSIONS.items()
    for extension in extensions
}

CANONICAL_MEDIA_TYPE: dict[str, str] = {
    PDF: "application/pdf",
    JPEG: "image/jpeg",
    PNG: "image/png",
    TIFF: "image/tiff",
    WEBP: "image/webp",
    DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class FileValidationError(Exception):
    """An upload that cannot be processed, with a reason fit to show a user.

    `code` is the machine-readable discriminator; the message is the human
    one. A UI that only had the message would have to string-match it to
    decide whether to offer "try a different file" or "your file is too big".
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ValidatedUpload:
    file_format: str
    media_type: str
    extension: str
    size_bytes: int
    sha256: str
    page_count: int | None = None

    @property
    def needs_ocr(self) -> bool:
        """PDFs are undecided at this point — whether they need OCR depends on
        their text layer, which the PDF service decides per page."""
        return self.file_format in RASTER_FORMATS


def detect_format(payload: bytes) -> str | None:
    """Identify a file from its leading bytes. Returns None for anything not
    in the supported set — including files that are perfectly valid, just not
    something this pipeline reads."""
    if len(payload) < 12:
        return None

    if payload[:5] == b"%PDF-":
        return PDF
    if payload[:3] == b"\xff\xd8\xff":
        return JPEG
    if payload[:8] == b"\x89PNG\r\n\x1a\n":
        return PNG
    if payload[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return TIFF
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return WEBP
    if payload[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        # DOCX and XLSX are both zip containers, indistinguishable this early.
        # Resolved by looking inside; see `_resolve_ooxml`.
        return _resolve_ooxml(payload)
    return None


def _resolve_ooxml(payload: bytes) -> str | None:
    """Tell a .docx from a .xlsx by the part names inside the zip.

    Scanning the raw bytes for the marker path is deliberate: opening the
    archive would mean parsing attacker-controlled zip structure during
    validation, before the file has earned that trust.
    """
    window = payload[:4096]
    if b"word/" in window:
        return DOCX
    if b"xl/" in window:
        return XLSX
    return None


def validate_upload(
    payload: bytes,
    filename: str,
    declared_media_type: str | None,
    max_bytes: int,
    allowed_formats: frozenset[str] | set[str] | None = None,
) -> ValidatedUpload:
    """Run the full validation ladder and describe the file, or raise.

    `allowed_formats` narrows the accepted set for a particular endpoint —
    the invoice upload accepts the five formats the brief names, while the
    general document upload also takes Office files.
    """
    extension = Path(filename or "").suffix.lower()
    permitted = frozenset(allowed_formats) if allowed_formats else frozenset(EXTENSIONS)

    if not payload:
        raise FileValidationError("EMPTY_FILE", "The uploaded file is empty.")

    if len(payload) > max_bytes:
        raise FileValidationError(
            "FILE_TOO_LARGE",
            f"File is {_human_size(len(payload))}; the limit is {_human_size(max_bytes)}.",
        )

    if extension not in EXTENSION_TO_FORMAT:
        raise FileValidationError(
            "UNSUPPORTED_EXTENSION",
            f"'{extension or filename}' is not a supported file type. "
            f"Accepted: {_describe(permitted)}.",
        )

    actual_format = detect_format(payload)
    if actual_format is None:
        raise FileValidationError(
            "UNRECOGNISED_CONTENT",
            "The file contents do not match any supported format. It may be "
            "corrupted, or renamed from an unsupported type.",
        )

    if actual_format not in permitted:
        raise FileValidationError(
            "UNSUPPORTED_FORMAT",
            f"{actual_format.upper()} files are not accepted here. "
            f"Accepted: {_describe(permitted)}.",
        )

    # The extension is checked against the *detected* format, not the other
    # way round: an invoice.pdf that is really a JPEG should be rejected with
    # a message about the mismatch, not silently processed as whichever the
    # next stage happens to guess.
    if EXTENSION_TO_FORMAT[extension] != actual_format:
        raise FileValidationError(
            "CONTENT_EXTENSION_MISMATCH",
            f"The file is named '{extension}' but its contents are "
            f"{actual_format.upper()}. Rename it or upload the original file.",
        )

    if declared_media_type:
        normalised = declared_media_type.split(";")[0].strip().lower()
        # A wrong Content-Type is logged, not fatal: browsers and mobile
        # clients get this wrong routinely, and the bytes have already
        # settled the question authoritatively.
        if normalised and normalised not in MEDIA_TYPES[actual_format]:
            logger.info(
                "upload.declared_media_type_mismatch declared=%s detected=%s filename=%s",
                normalised, actual_format, filename,
            )

    page_count = _verify_readable(payload, actual_format)

    return ValidatedUpload(
        file_format=actual_format,
        media_type=CANONICAL_MEDIA_TYPE[actual_format],
        extension=extension,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        page_count=page_count,
    )


def _verify_readable(payload: bytes, file_format: str) -> int | None:
    """Open the file for real, so a truncated download fails at upload rather
    than an hour later in a worker. Returns the page/frame count where the
    format has one."""
    if file_format == PDF:
        return _verify_pdf(payload)
    if file_format in RASTER_FORMATS:
        return _verify_image(payload, file_format)
    return None  # Office formats are verified when they are read; a zip that
    # opens is not evidence the document inside is valid, and parsing it here
    # would duplicate the extractor.


def _verify_pdf(payload: bytes) -> int:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - PyMuPDF is a hard dependency
        raise FileValidationError("READER_UNAVAILABLE", "PDF support is not installed.") from exc

    try:
        with fitz.open(stream=payload, filetype="pdf") as document:
            if document.needs_pass:
                raise FileValidationError(
                    "PASSWORD_PROTECTED",
                    "This PDF is password-protected. Remove the password and upload it again.",
                )
            page_count = document.page_count
            if page_count < 1:
                raise FileValidationError("EMPTY_PDF", "This PDF contains no pages.")
            # Touching the first page forces the object tree to be parsed;
            # a truncated file opens happily and only fails on access.
            document.load_page(0)

            if document.is_repaired and not _has_recoverable_content(document):
                # MuPDF silently rebuilds a damaged cross-reference table, so
                # a truncated download *opens* and reports a plausible page
                # count — it just has nothing left inside. Repair alone is not
                # grounds for rejection (real generators emit files needing
                # it), but repair plus no recoverable content is exactly what
                # a half-finished upload looks like, and letting it through
                # costs a worker an OCR pass to produce an empty invoice.
                raise FileValidationError(
                    "CORRUPT_PDF",
                    "This PDF is damaged — its content could not be recovered. "
                    "It may have been truncated during upload; try again with the original file.",
                )
            return page_count
    except FileValidationError:
        raise
    except Exception as exc:
        raise FileValidationError("CORRUPT_PDF", f"This PDF could not be read: {exc}") from exc


def _has_recoverable_content(document) -> bool:
    """Whether any page still holds text or an image.

    Both are checked because either one alone is a normal invoice: a
    generated PDF has text and no images, a scanned one has an image and no
    text. A page with neither is not an invoice this pipeline can process.
    Only the first few pages are examined — a document whose opening pages
    are all empty is not worth an OCR pass regardless of what follows.
    """
    for index in range(min(document.page_count, 5)):
        try:
            page = document.load_page(index)
            if page.get_text().strip():
                return True
            if page.get_images():
                return True
        except Exception:
            continue
    return False


def _verify_image(payload: bytes, file_format: str) -> int:
    """Verify a raster image and count its frames.

    Pillow rather than OpenCV: `cv2.imdecode` reads only the first frame of a
    multi-page TIFF and reports nothing about the rest, and multi-page TIFF is
    precisely how departmental scanners emit a stapled invoice.
    """
    import io

    try:
        from PIL import Image, ImageFile
    except ImportError as exc:  # pragma: no cover - Pillow ships with the OCR stack
        raise FileValidationError("READER_UNAVAILABLE", "Image support is not installed.") from exc

    # A truncated JPEG must fail here rather than being silently padded with
    # grey, which is Pillow's default and would send a half-image to OCR.
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()  # checks structure, but consumes the file object
        with Image.open(io.BytesIO(payload)) as image:
            frames = getattr(image, "n_frames", 1)
            image.load()   # forces full decode; verify() alone misses truncation
            width, height = image.size
    except FileValidationError:
        raise
    except Exception as exc:
        raise FileValidationError(
            "CORRUPT_IMAGE", f"This {file_format.upper()} image could not be read: {exc}"
        ) from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous

    if width < 100 or height < 100:
        raise FileValidationError(
            "IMAGE_TOO_SMALL",
            f"The image is {width}x{height} pixels — too small to contain a readable invoice.",
        )
    return int(frames)


def _describe(formats: frozenset[str] | set[str]) -> str:
    return ", ".join(sorted(file_format.upper() for file_format in formats))


def _human_size(size_bytes: int) -> str:
    megabytes = size_bytes / (1024 * 1024)
    if megabytes >= 1:
        return f"{megabytes:.1f} MB"
    return f"{size_bytes / 1024:.0f} KB"


def sha256_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_storage_key(prefix: str, document_number: str, filename: str) -> str:
    """Build a storage key that cannot escape the storage root.

    The uploaded filename is preserved for the user's benefit but stripped of
    every path component and unusual character first: it is attacker-supplied
    text that is about to become a filesystem path, and '../' in a filename is
    the oldest trick there is.
    """
    stem = Path(filename or "upload").name
    cleaned = "".join(character if character.isalnum() or character in "._- " else "_" for character in stem)
    cleaned = cleaned.strip(". ") or "upload"
    return f"{prefix}/{document_number}/{cleaned[:120]}"
