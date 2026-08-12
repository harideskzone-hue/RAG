import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.platform.config.config import config


def init_tracer(service_name: str = "vista_agentic_ai") -> trace.Tracer:
    """
    Initializes the OpenTelemetry Tracer with the configured Exporter.
    """
    resource = Resource.create(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)
    
    if config.telemetry_exporter == "console":
        exporter = ConsoleSpanExporter()
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
    elif config.telemetry_exporter == "otlp":
        exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
    
    # Register the global provider
    trace.set_tracer_provider(provider)
    
    return trace.get_tracer(service_name)

# Expose a default tracer for immediate use in decorators
tracer = trace.get_tracer("vista_agentic_ai")
