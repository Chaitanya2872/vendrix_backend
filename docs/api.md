# API Documentation

This document summarizes the REST API exposed by the IoTIQ Vendor Management backend.

## Base URL

- Local development: http://localhost:8000/api/v1
- Interactive Swagger UI: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/api/v1/openapi.json

## Authentication

Most endpoints require a bearer token. Authenticate with:

- POST /auth/login
- POST /auth/register (ADMIN only)
- GET /auth/me

Use the returned token in the Authorization header:

```http
Authorization: Bearer <access_token>
```

## Common response conventions

- Success responses are JSON payloads unless noted otherwise.
- `204 No Content` is returned for successful delete operations.
- Typical error responses include `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `409 Conflict`, `422 Unprocessable Entity`, and `415 Unsupported Media Type`.

## Endpoint reference

### Authentication

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| POST | /auth/login | Sign in and receive an access token | Body: `email`, `password` |
| POST | /auth/register | Create a new user account | ADMIN only; body includes `full_name`, `role` |
| GET | /auth/me | Return the signed-in user profile | Requires authentication |

### Users

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /users | List users | ADMIN only |

### Vendors

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /vendors | List vendors | Supports `limit` and `offset` |
| POST | /vendors | Create a vendor | Requires `ADMIN` or `OPERATOR` |
| GET | /vendors/{vendor_id} | Fetch a vendor | Requires authentication |
| PATCH | /vendors/{vendor_id} | Update a vendor | Requires `ADMIN` or `OPERATOR` |
| DELETE | /vendors/{vendor_id} | Delete a vendor | ADMIN only |

Common vendor fields:
- `vendor_code`
- `legal_name`
- `gstin`
- `category`
- `status`
- `phone`
- `email`
- `address`
- `bank_details`

### Vendor categories

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /vendor-categories | List categories | Standard CRUD |
| POST | /vendor-categories | Create a category | ADMIN only |
| GET | /vendor-categories/{item_id} | Fetch one category | Requires authentication |
| PATCH | /vendor-categories/{item_id} | Update a category | ADMIN only |
| DELETE | /vendor-categories/{item_id} | Delete a category | ADMIN only |

### Vehicles

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /vehicles | List vehicles | Standard CRUD |
| POST | /vehicles | Create a vehicle | Requires `ADMIN` or `OPERATOR` |
| GET | /vehicles/{item_id} | Fetch a vehicle | Requires authentication |
| PATCH | /vehicles/{item_id} | Update a vehicle | Requires `ADMIN` or `OPERATOR` |
| DELETE | /vehicles/{item_id} | Delete a vehicle | ADMIN only |

### Drivers

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /drivers | List drivers | Standard CRUD |
| POST | /drivers | Create a driver | Requires `ADMIN` or `OPERATOR` |
| GET | /drivers/{item_id} | Fetch a driver | Requires authentication |
| PATCH | /drivers/{item_id} | Update a driver | Requires `ADMIN` or `OPERATOR` |
| DELETE | /drivers/{item_id} | Delete a driver | ADMIN only |

### Invoices

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /invoices | List invoices | Standard CRUD |
| POST | /invoices | Create an invoice | Requires `ADMIN`, `OPERATOR`, or `FINANCE` |
| GET | /invoices/{item_id} | Fetch an invoice | Requires authentication |
| PATCH | /invoices/{item_id} | Update an invoice | Requires `ADMIN`, `OPERATOR`, or `FINANCE` |
| DELETE | /invoices/{item_id} | Delete an invoice | ADMIN only |
| POST | /invoices/{invoice_id}/submit | Submit a draft invoice for approval | Changes status to `PENDING_APPROVAL` |

### Payments

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /payments | List payments | Standard CRUD |
| POST | /payments | Create a payment | Requires `ADMIN` or `FINANCE` |
| GET | /payments/{item_id} | Fetch a payment | Requires authentication |
| PATCH | /payments/{item_id} | Update a payment | Requires `ADMIN` or `FINANCE` |
| DELETE | /payments/{item_id} | Delete a payment | ADMIN only |

### Purchases

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /purchases | List purchases | Supports `status`, `limit`, and `offset` |
| POST | /purchases | Create a purchase | Requires `ADMIN` or `OPERATOR` |
| GET | /purchases/{purchase_id} | Fetch a purchase | Requires authentication |
| PATCH | /purchases/{purchase_id} | Update a purchase | Requires `ADMIN` or `OPERATOR` |

### Deliveries

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /deliveries | List deliveries | Supports `status`, `limit`, and `offset` |
| POST | /deliveries | Create a delivery | Requires `ADMIN` or `OPERATOR` |
| GET | /deliveries/{delivery_id} | Fetch a delivery | Requires authentication |
| PATCH | /deliveries/{delivery_id} | Update a delivery | Requires `ADMIN` or `OPERATOR` |

### Vehicle entries (gate register)

Inward and outward vehicle movements across a site gate. One record covers a
whole visit: `direction` says whether the vehicle is bringing material in
(`INWARD`) or taking it out (`OUTWARD`), `entry_at` is when it arrived and
`exit_at` when it left. A visit with no `exit_at` has status `IN_PREMISES`.

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /vehicle-entries | List movements | Filters: `direction`, `status`, `purpose`, `gate`, `vendor_id`, `vehicle_id`, `registration_number`, `entry_from`, `entry_to`, `q`, `limit`, `offset` |
| POST | /vehicle-entries | Record a gate-in | Requires `ADMIN`, `OPERATOR`, or `SECURITY` |
| GET | /vehicle-entries/on-premises | Vehicles currently inside, oldest first | Optional `gate` |
| GET | /vehicle-entries/summary | Gate dashboard counts for the current UTC day | Requires authentication |
| GET | /vehicle-entries/{entry_id} | Fetch one movement | Requires authentication |
| PATCH | /vehicle-entries/{entry_id} | Correct or cancel a movement | Requires `ADMIN`, `OPERATOR`, or `SECURITY` |
| POST | /vehicle-entries/{entry_id}/exit | Sign the vehicle out | Requires `ADMIN`, `OPERATOR`, or `SECURITY` |
| GET | /vehicle-entries/vehicle/{registration_number}/history | Visit history for one plate | Reports whether it is inside now |

Entry fields:
- `direction` — `INWARD` or `OUTWARD`
- `purpose` — `DELIVERY`, `PICKUP`, `SERVICE`, `TRANSFER`, `VISITOR`, `OTHER`
- `registration_number` — normalised to uppercase alphanumerics; required unless `vehicle_id` is given
- `vehicle_id`, `vendor_id`, `driver_id`, `purchase_id`, `delivery_id` — optional links
- `driver_name`, `driver_phone` — the person on the pass, whether or not they are a registered driver
- `gate`, `document_reference`, `material_description`, `remarks`
- `gross_weight`, `tare_weight` — weighbridge readings; `net_weight` is derived as gross minus tare
- `entry_at` — defaults to the time the request is handled
- `capture_method` — `MANUAL` or `ANPR`

Behaviour worth knowing:
- A plate that already has an open visit returns `409` naming the open entry number.
- Signing a vehicle out twice returns `409`; an exit earlier than the entry returns `422`.
- Gross below tare returns `422` rather than storing a negative net weight.
- A plate that is not in the fleet registry is accepted and reported with `vehicle_registered: false`.
- Gate-in and gate-out are written to the audit log as `GATE_IN` and `GATE_OUT`.

### Approvals

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /approvals | List approvals | Requires `ADMIN`, `APPROVER`, or `FINANCE` |
| POST | /approvals/{approval_id}/decision | Approve or reject an approval | Body: `decision`, `comment` |

### Documents

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /documents | List documents | Requires authentication |
| POST | /documents | Upload a document | Form field `file`; query param `document_type` |
| GET | /documents/{document_id} | Fetch document metadata | Requires authentication |
| GET | /documents/{document_id}/download | Download stored file | Requires authentication |
| GET | /documents/{document_id}/preview | Preview supported document formats | Requires authentication |
| DELETE | /documents/{document_id} | Delete a document | Requires authentication |
| POST | /documents/{document_id}/review | Confirm extracted fields | Body: `fields` |

Supported upload extensions:
- `.pdf`
- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.docx`
- `.xlsx`

### ANPR

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| POST | /anpr/lookup | Lookup a vehicle by registration number | Requires authentication |
| POST | /anpr/recognize | Recognize a plate from an uploaded image | Requires authentication |

Both ANPR responses also carry the gate context for the plate — `on_premises`,
the `open_entry` if there is one, and a `suggested_action` of `GATE_IN` or
`GATE_OUT` — so one camera read can drive either half of a visit.

### Reports

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /reports/summary | Get summary metrics | Requires `ADMIN`, `FINANCE`, or `APPROVER` |

### Mobile

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /mobile/dashboard | Return dashboard metrics for the current user | Requires authentication |

### Audit logs

| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| GET | /audit-logs | List audit entries | ADMIN only |

## Notes

- The application seeds a default administrator account on first startup.
- Document processing and OCR tasks are triggered asynchronously after upload.
- The backend uses SQLite by default in local development, so the API is suitable for lightweight prototyping and testing.
