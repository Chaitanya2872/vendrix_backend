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
