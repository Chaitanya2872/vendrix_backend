"""Gate register API.

The cases here are the ones a guard can actually hit on a shift: the same
truck presented twice, an exit typed with yesterday's time, a plate nobody
onboarded, and the weighbridge readings arriving one at a time.
"""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.models import Vehicle, VehicleEntry, Vendor


def plate() -> str:
    """A distinct, valid-length registration for each test."""
    return f"KA01AB{uuid4().int % 10000:04d}"


@pytest.fixture
def vendor(db) -> Vendor:
    record = Vendor(vendor_code=f"V{uuid4().hex[:8]}", legal_name="Gate Test Supplies")
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@pytest.fixture
def vehicle(db, vendor) -> Vehicle:
    record = Vehicle(
        vendor_id=vendor.id,
        registration_number=plate(),
        vehicle_type="TRUCK",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@pytest.fixture(autouse=True)
def _clean_entries(db):
    """Each test starts with an empty register; the summary and on-premises
    endpoints are counts over the whole table and would otherwise see rows
    left by their neighbours."""
    yield
    db.query(VehicleEntry).delete()
    db.commit()


def test_records_an_inward_entry_and_links_the_registered_vehicle(client, vehicle, vendor):
    response = client.post(
        "/api/v1/vehicle-entries",
        json={
            "direction": "INWARD",
            "registration_number": vehicle.registration_number,
            "purpose": "DELIVERY",
            "driver_name": "R. Kumar",
            "gate": "Gate 1",
            "document_reference": "DC-8891",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["entry_number"].startswith("GE-")
    assert body["direction"] == "INWARD"
    assert body["status"] == "IN_PREMISES"
    # The plate was recognised, so the vendor came along with it.
    assert body["vehicle_id"] == vehicle.id
    assert body["vendor_id"] == vendor.id
    assert body["vehicle_registered"] is True
    assert body["exit_at"] is None
    assert body["duration_minutes"] is None


def test_accepts_a_vehicle_that_is_not_in_the_fleet_registry(client):
    response = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "OUTWARD", "registration_number": plate(), "purpose": "PICKUP"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["vehicle_id"] is None
    assert body["vehicle_registered"] is False


def test_registration_number_is_normalised(client):
    raw = "ka 05-mn 4321"
    response = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": raw},
    )
    assert response.status_code == 201, response.text
    assert response.json()["registration_number"] == "KA05MN4321"


def test_a_vehicle_already_inside_cannot_be_signed_in_again(client):
    registration = plate()
    first = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": registration},
    )
    assert first.status_code == 201

    second = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": registration},
    )
    assert second.status_code == 409
    # The guard needs to know *which* pass to close, not just that one exists.
    assert first.json()["entry_number"] in second.json()["detail"]


def test_a_vehicle_can_return_after_it_has_been_signed_out(client):
    registration = plate()
    first = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": registration},
    ).json()
    assert client.post(f"/api/v1/vehicle-entries/{first['id']}/exit", json={}).status_code == 200

    again = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": registration},
    )
    assert again.status_code == 201


def test_exit_closes_the_visit_and_reports_the_duration(client):
    entry_at = datetime.now(timezone.utc) - timedelta(hours=2)
    entry = client.post(
        "/api/v1/vehicle-entries",
        json={
            "direction": "INWARD",
            "registration_number": plate(),
            "entry_at": entry_at.isoformat(),
        },
    ).json()

    response = client.post(
        f"/api/v1/vehicle-entries/{entry['id']}/exit",
        json={"exit_at": (entry_at + timedelta(minutes=90)).isoformat()},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "COMPLETED"
    assert body["exit_at"] is not None
    assert body["duration_minutes"] == 90


def test_exit_cannot_predate_the_entry(client):
    entry = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": plate()},
    ).json()

    response = client.post(
        f"/api/v1/vehicle-entries/{entry['id']}/exit",
        json={"exit_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
    )
    assert response.status_code == 422


def test_a_vehicle_cannot_be_signed_out_twice(client):
    entry = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": plate()},
    ).json()
    assert client.post(f"/api/v1/vehicle-entries/{entry['id']}/exit", json={}).status_code == 200
    assert client.post(f"/api/v1/vehicle-entries/{entry['id']}/exit", json={}).status_code == 409


def test_net_weight_appears_once_both_readings_exist(client):
    entry = client.post(
        "/api/v1/vehicle-entries",
        json={
            "direction": "INWARD",
            "registration_number": plate(),
            "gross_weight": 18500.5,
        },
    ).json()
    # Loaded on the way in; nothing to subtract from yet.
    assert entry["gross_weight"] == 18500.5
    assert entry["net_weight"] is None

    closed = client.post(
        f"/api/v1/vehicle-entries/{entry['id']}/exit",
        json={"tare_weight": 8500.5},
    ).json()
    assert closed["net_weight"] == 10000.0


def test_gross_below_tare_is_rejected(client):
    response = client.post(
        "/api/v1/vehicle-entries",
        json={
            "direction": "INWARD",
            "registration_number": plate(),
            "gross_weight": 4000,
            "tare_weight": 9000,
        },
    )
    assert response.status_code == 422


def test_direction_and_purpose_are_constrained(client):
    response = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "SIDEWAYS", "registration_number": plate()},
    )
    assert response.status_code == 422


def test_unknown_vendor_reference_is_rejected(client):
    response = client.post(
        "/api/v1/vehicle-entries",
        json={
            "direction": "INWARD",
            "registration_number": plate(),
            "vendor_id": str(uuid4()),
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Vendor not found"


def test_list_filters_by_direction_and_status(client):
    inward = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": plate()},
    ).json()
    client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "OUTWARD", "registration_number": plate()},
    )
    client.post(f"/api/v1/vehicle-entries/{inward['id']}/exit", json={})

    outward = client.get("/api/v1/vehicle-entries", params={"direction": "OUTWARD"}).json()
    assert outward["total"] == 1
    assert outward["items"][0]["direction"] == "OUTWARD"

    completed = client.get("/api/v1/vehicle-entries", params={"status": "COMPLETED"}).json()
    assert completed["total"] == 1
    assert completed["items"][0]["id"] == inward["id"]


def test_search_matches_the_document_reference(client):
    client.post(
        "/api/v1/vehicle-entries",
        json={
            "direction": "INWARD",
            "registration_number": plate(),
            "document_reference": "DC-77120",
        },
    )
    found = client.get("/api/v1/vehicle-entries", params={"q": "dc-771"}).json()
    assert found["total"] == 1


def test_on_premises_lists_only_open_visits_oldest_first(client):
    older = client.post(
        "/api/v1/vehicle-entries",
        json={
            "direction": "INWARD",
            "registration_number": plate(),
            "entry_at": (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),
        },
    ).json()
    newer = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": plate()},
    ).json()
    gone = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "OUTWARD", "registration_number": plate()},
    ).json()
    client.post(f"/api/v1/vehicle-entries/{gone['id']}/exit", json={})

    body = client.get("/api/v1/vehicle-entries/on-premises").json()
    assert [item["id"] for item in body["items"]] == [older["id"], newer["id"]]


def test_summary_counts_the_current_day(client, vehicle):
    client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": vehicle.registration_number},
    )
    outward = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "OUTWARD", "registration_number": plate()},
    ).json()
    client.post(f"/api/v1/vehicle-entries/{outward['id']}/exit", json={})

    body = client.get("/api/v1/vehicle-entries/summary").json()
    assert body["inward_today"] == 1
    assert body["outward_today"] == 1
    assert body["on_premises"] == 1
    assert body["completed_today"] == 1
    # The registered truck is the one still inside, so nothing unknown is.
    assert body["unregistered_on_premises"] == 0


def test_history_reports_whether_the_plate_is_inside(client):
    registration = plate()
    client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": registration},
    )
    body = client.get(f"/api/v1/vehicle-entries/vehicle/{registration}/history").json()
    assert body["on_premises"] is True
    assert body["open_entry"]["registration_number"] == registration
    assert body["total"] == 1


def test_update_corrects_a_visit_and_can_cancel_it(client):
    entry = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": plate()},
    ).json()

    response = client.patch(
        f"/api/v1/vehicle-entries/{entry['id']}",
        json={"document_reference": "INV-4410", "status": "CANCELLED"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["document_reference"] == "INV-4410"
    assert response.json()["status"] == "CANCELLED"

    # A cancelled visit is not a vehicle that is still inside.
    assert client.get("/api/v1/vehicle-entries/on-premises").json()["total"] == 0


def test_a_cancelled_visit_cannot_be_signed_out(client):
    entry = client.post(
        "/api/v1/vehicle-entries",
        json={"direction": "INWARD", "registration_number": plate()},
    ).json()
    client.patch(f"/api/v1/vehicle-entries/{entry['id']}", json={"status": "CANCELLED"})

    assert client.post(f"/api/v1/vehicle-entries/{entry['id']}/exit", json={}).status_code == 409


def test_unknown_entry_is_a_404(client):
    assert client.get(f"/api/v1/vehicle-entries/{uuid4()}").status_code == 404
