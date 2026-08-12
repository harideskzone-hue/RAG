import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.platform.logging.logger import get_logger

logger = get_logger("vista.api")

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs incoming requests and their latency.
    Runs after TracingMiddleware so execution_id is available.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_host=request.client.host if request.client else None
        )
        
        try:
            response = await call_next(request)
            process_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                latency_ms=process_time_ms
            )
            return response
        except Exception as e:
            process_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                latency_ms=process_time_ms
            )
            raise e
