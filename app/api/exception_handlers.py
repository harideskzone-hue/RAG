from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.platform.errors.base import VistaError
from app.platform.errors.memory import MemoryError
from app.platform.errors.repository import RepositoryError
from app.platform.errors.security import SecurityError
from app.platform.errors.service import ServiceError
from app.platform.errors.tool import ToolError
from app.platform.errors.validation import ValidationError
from app.platform.errors.workflow import WorkflowError


def add_exception_handlers(app: FastAPI):
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        return JSONResponse(status_code=400, content={"error": "Validation Error", "detail": str(exc)})

    @app.exception_handler(SecurityError)
    async def security_error_handler(request: Request, exc: SecurityError):
        return JSONResponse(status_code=403, content={"error": "Security Error", "detail": str(exc)})
        
    @app.exception_handler(ToolError)
    @app.exception_handler(RepositoryError)
    @app.exception_handler(ServiceError)
    @app.exception_handler(WorkflowError)
    @app.exception_handler(MemoryError)
    async def internal_component_error_handler(request: Request, exc: VistaError):
        return JSONResponse(status_code=500, content={"error": "Internal Component Error", "detail": str(exc)})

    @app.exception_handler(VistaError)
    async def generic_vista_error_handler(request: Request, exc: VistaError):
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "detail": str(exc)})
