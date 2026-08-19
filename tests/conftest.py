"""Shared test fixtures.

The database URL and storage path are redirected to throwaway locations
*before* any `app.` module is imported, because app.core.config builds its
Settings at import time and app.db.session binds an engine off the back of
it. Importing the app first would permanently point the tests at the real
iotiq.db and ./storage.
"""
import os
import tempfile
from pathlib import Path
from uuid import uuid4

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="iotiq-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_ROOT / 'test.db').as_posix()}"
os.environ["STORAGE_PATH"] = str(_TMP_ROOT / "storage")
os.environ["CELERY_ENABLED"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.common.dependencies import current_user  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Document, User  # noqa: E402

from tests.invoice_samples import write_sample  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _schema():
    """One schema for the whole session; individual tests clean up their own
    rows rather than paying to rebuild every table each time."""
    Path(settings.storage_path).mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def user(db) -> User:
    """A persisted, active user — uploads need a real FK target."""
    existing = db.query(User).filter_by(email="tester@example.com").first()
    if existing:
        return existing
    record = User(
        email="tester@example.com",
        full_name="Test User",
        password_hash="not-a-real-hash",
        role="OPERATOR",
        is_active=True,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@pytest.fixture
def client(user):
    """Authenticated client. The JWT path itself is covered separately by
    test_documents_api.test_upload_requires_authentication, which uses the
    app without this override."""
    app.dependency_overrides[current_user] = lambda: user
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample(tmp_path):
    """Writes one of the generated invoice samples and returns its path.

    Usage: `sample("text_pdf")` — see tests/invoice_samples.py for the kinds.
    """
    def _make(kind: str, name: str | None = None) -> Path:
        return write_sample(kind, tmp_path, name)

    return _make


@pytest.fixture
def stored_document(db, user, sample):
    """A Document row whose file really exists in the configured storage,
    which is what the parsing pipeline expects to find."""
    def _make(kind: str, document_type: str = "INVOICE") -> Document:
        source = sample(kind)
        # Unique per call, mirroring the upload endpoint's own key format.
        # Without the uuid, two tests asking for the same sample kind collide
        # on the object_key unique constraint — each passes alone and the pair
        # fails, which reads as a flaky test rather than a fixture bug.
        key = f"{user.id}/{uuid4().hex[:8]}_{source.name}"
        destination = Path(settings.storage_path) / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

        record = Document(
            filename=source.name,
            object_key=key,
            content_type="application/octet-stream",
            document_type=document_type,
            owner_id=user.id,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    return _make


@pytest.fixture
def seeded_document(db, user) -> Document:
    """A Document row with a document number, but no file on disk.

    For tests about run tracking and status reporting, where the pipeline
    never runs and the file would only be dead weight.
    """
    record = Document(
        filename="invoice.pdf",
        object_key=f"invoices/{uuid4().hex[:8]}/invoice.pdf",
        content_type="application/pdf",
        document_type="INVOICE",
        owner_id=user.id,
        document_number=f"DOC-2026-{uuid4().int % 1000000:06d}",
        file_format="pdf",
        size_bytes=1024,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
