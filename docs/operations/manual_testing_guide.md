# VISTA AI End-to-End Manual Testing Guide

This guide provides a comprehensive step-by-step process to manually test the VISTA AI Agent pipeline. It verifies that all components—video ingestion, metadata extraction, RAG retrieval, multi-agent orchestration, reasoning, memory, and response generation—function together seamlessly.

---

## Prerequisites

1. **Local Stack Running**: Ensure the environment is up and healthy.
   ```bash
   make start
   make smoke
   ```
2. **Tools Required**: `curl` (or Postman) and a sample `.mp4` video (e.g., `sample_cctv.mp4` showing a person in a red shirt walking).
3. **Environment**: Ensure your `.env.local` contains valid API keys for the Gemini VLM adapter (`GEMINI_API_KEY`).

---

## Phase 1: Video Ingestion & Indexing

**Objective**: Verify the `VideoService`, `S3Tool`, and `VectorTool` successfully upload, analyze, and index a new video.

### 1. Upload Video via API
```bash
curl -X POST "http://localhost:8000/api/v1/video/upload" \
  -H "Authorization: Bearer <ADMIN_JWT_TOKEN>" \
  -F "file=@sample_cctv.mp4" \
  -F "camera_id=cam_1" \
  -F "location=front_entrance"
```

### 2. Validation Checkpoints
- **Response**: Should return HTTP 202 Accepted with a `task_id` and `video_id`.
- **S3 Bucket**: Verify the file exists in the MinIO `vista-videos` bucket.
- **Postgres Database**: Verify the `videos` table contains a new row with `status="processing"`.

### 3. Asynchronous Processing Verification
Wait ~30 seconds, then check the video status:
```bash
curl -X GET "http://localhost:8000/api/v1/video/status/{video_id}" \
  -H "Authorization: Bearer <ADMIN_JWT_TOKEN>"
```
- **Success Criteria**: `status` transitions to `completed`.
- **Milvus Verification**: The vector store should now contain embeddings representing the events/persons detected in the video.

---

## Phase 2: Core Agentic AI Pipeline Tests

**Objective**: Test natural language understanding, multi-agent orchestration, memory, and response generation.

We will simulate a chat session. Note the `conversation_id` you use to test memory in Scenario 2.

### Scenario 1: Complex Multi-Agent Reasoning (Person Search)
**User Query**: *"Find a person wearing a red shirt near the front entrance from today."*

**Execution**:
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find a person wearing a red shirt near the front entrance from today.",
    "conversation_id": "test-session-001"
  }'
```

**What to Watch For (Orchestration Trace)**:
1. **Intent Agent**: Classifies query as `PERSON_SEARCH`.
2. **Planner Agent**: Generates an ExecutionPlan targeting `MetadataAgent`, `VectorAgent`, and `ConfidenceAgent`.
3. **Metadata Agent**: Executes `PostgresTool` to filter video IDs matching `location="front_entrance"` and `date="today"`.
4. **Vector Agent**: Executes `MilvusTool` to perform a semantic search for "person wearing red shirt" restricted to the filtered video IDs.
5. **Confidence Agent**: Evaluates the retrieved evidence bundles, scoring them > 0.8.
6. **Supervisor**: Aggregates the high-confidence evidence and generates a natural language response.

**Success Criteria**:
- HTTP 200 OK.
- Response narrative identifies the video clip and timestamp where the person was seen.
- `Trace ID` is returned in the response headers.

### Scenario 2: Memory & Contextual Follow-up
**User Query**: *"Did they have a backpack?"*

**Execution**:
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Did they have a backpack?",
    "conversation_id": "test-session-001"
  }'
```

**What to Watch For**:
1. **Memory Load**: The Supervisor retrieves the Redis checkpoint for `test-session-001`, injecting the "red shirt at front entrance" context into the new prompt.
2. **Intent Agent**: Resolves the pronoun "they".
3. **Video Agent (VLM)**: Executes `S3Tool` and `GeminiAdapter` to visually inspect the specific frames retrieved in Scenario 1 to check for a backpack.

**Success Criteria**:
- The agent correctly understands "they" refers to the person in the red shirt from the previous query.
- Returns a factual yes/no based on the VLM analysis of the frames.

### Scenario 3: Event Correlation (Report Generation)
**User Query**: *"Generate an incident report for all anomalies detected on camera 1 this week."*

**Execution**:
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Generate an incident report for all anomalies detected on camera 1 this week.",
    "conversation_id": "test-session-002"
  }'
```

**Success Criteria**:
- **Report Agent**: Triggers report generation.
- Response includes a download URI (e.g., `/api/v1/reports/download/{report_id}`).
- The generated PDF/Markdown report successfully correlates the events.

---

## Phase 3: Reliability & Error Handling

**Objective**: Verify graceful degradation when infrastructure fails.

### Test: Database Timeout (Milvus Offline)
1. Stop the Milvus container to simulate an outage:
   ```bash
   docker stop vista-milvus
   ```
2. Re-run the Person Search query (Scenario 1).

**Expected Behavior**:
- The `VectorAgent` attempts to query Milvus and catches a connection timeout.
- The Supervisor's Failure Handler catches the exception, triggering the retry policy (max 3 retries).
- The Supervisor transitions to Graceful Degradation mode.
- **Success Criteria**: The API returns HTTP 200 (NOT 500) with a partial response indicating: *"I found 2 videos matching the location in the metadata, but the semantic search system is currently offline, so I cannot isolate the exact frames."*

3. Restart Milvus:
   ```bash
   docker start vista-milvus
   ```

---

## Phase 4: Observability Verification

**Objective**: Ensure the entire trace was captured.

1. Open Jaeger UI at `http://localhost:16686`.
2. Search for traces originating from the `vista-api` service within the last 15 minutes.
3. Select the trace for the Person Search query.

**Success Criteria**:
- A single parent trace spans the entire request.
- Child spans exist for: `Supervisor`, `PlannerAgent`, `MetadataAgent`, `VectorAgent`, and `MilvusTool`.
- The database queries (SQL/Vector) are visible inside the tool spans.
- The `conversation_id` and `execution_id` are attached as tags to every span.

---

## Troubleshooting

- **Agent returns "I don't know"**: Check the Confidence Engine logs. If the threshold falls below 0.3, the query is rejected. Adjust the prompt or ensure the video actually contains the queried subject.
- **VLM Errors (HTTP 429/500)**: Check `docker logs vista-api`. Ensure your `GEMINI_API_KEY` is valid and hasn't hit rate limits.
- **Redis Checkpoint Fails**: Context isn't retained between questions. Ensure `vista-redis` is healthy and the `conversation_id` string exactly matches the previous request.
