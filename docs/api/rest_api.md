# REST API

VISTA AI exposes a standard RESTful interface for synchronous workloads.

## `POST /api/v1/chat`

Submit a query to the agent network.

**Request:**
```json
{
  "query": "Is camera 5 online?",
  "conversation_id": "conv-1234"
}
```

**Response (200 OK):**
```json
{
  "response": "Camera 5 is currently online.",
  "confidence": 1.0,
  "evidence_count": 1
}
```

## Authentication
All endpoints require a Bearer JWT token in the `Authorization` header. See [Authentication](authentication.md) for details.
