import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.platform.tracing.context import set_execution_id


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects a trace_id and execution_id into every request.
    This runs first in the middleware stack.
    """
    async def dispatch(self, request: Request, call_next):
        # Generate a unique execution ID for this request
        exec_id = str(uuid.uuid4())
        set_execution_id(exec_id)
        
        # Attach it to the request state so logging/routes can access it easily if needed
        request.state.execution_id = exec_id

        # OTEL ASGI instrumentation handles the span creation natively if we use opentelemetry-instrumentation-fastapi,
        # but we also set our custom ContextVar for downstream structlog injection.
        
        response = await call_next(request)
        response.headers["X-Execution-ID"] = exec_id
        return response
