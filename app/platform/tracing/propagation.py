from opentelemetry import trace


def inject_trace_context_to_vars():
    """
    Extracts the current OpenTelemetry span context and can be used 
    to populate custom contextvars if needed.
    """
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        # trace_id = format(ctx.trace_id, "032x")
        # span_id = format(ctx.span_id, "016x")
        # Structlog will natively extract OTEL contexts using its own processors
        # or we can manually map them here if we choose to.
