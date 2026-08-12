from opentelemetry import trace

from app.platform.tracing.context import (
    get_conversation_id,
    get_execution_id,
    get_user_id,
)


def inject_context_processor(logger, log_method, event_dict):
    """
    Structlog processor that injects standard observability context into every log entry.
    """
    # 1. OpenTelemetry Trace Context
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
        
    # 2. Business Correlation IDs from ContextVars
    execution_id = get_execution_id()
    if execution_id:
        event_dict["execution_id"] = execution_id
        
    conversation_id = get_conversation_id()
    if conversation_id:
        event_dict["conversation_id"] = conversation_id
        
    user_id = get_user_id()
    if user_id:
        event_dict["user_id"] = user_id
        
    return event_dict
