import functools
import inspect

from opentelemetry import trace

from app.platform.tracing.context import (
    get_conversation_id,
    get_execution_id,
    get_user_id,
)
from app.platform.tracing.tracer import tracer


def trace_layer(name: str, kind: str = "internal"):
    """
    Decorator to automatically instrument a function with an OpenTelemetry span,
    injecting relevant business context from contextvars.
    
    Args:
        name: Name of the span (e.g., "VideoService", "MetadataAgent")
        kind: Kind of the layer (e.g., "agent", "service", "tool", "repository")
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("layer.kind", kind)
                
                if get_execution_id():
                    span.set_attribute("vista.execution_id", get_execution_id())
                if get_conversation_id():
                    span.set_attribute("vista.conversation_id", get_conversation_id())
                if get_user_id():
                    span.set_attribute("vista.user_id", get_user_id())
                    
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise e
                    
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("layer.kind", kind)
                
                if get_execution_id():
                    span.set_attribute("vista.execution_id", get_execution_id())
                if get_conversation_id():
                    span.set_attribute("vista.conversation_id", get_conversation_id())
                    
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise e

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator
