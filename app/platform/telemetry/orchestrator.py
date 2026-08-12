from app.platform.config.config import config
from app.platform.logging.logger import init_logging
from app.platform.tracing.tracer import init_tracer


def init_telemetry():
    """
    Initializes all telemetry components: Tracing, Logging, Metrics.
    To be called once at application startup.
    """
    # 1. Initialize Structured Logging
    init_logging(json_format=config.json_logs)
    
    # 2. Initialize Tracing if enabled
    if config.enable_tracing:
        init_tracer(config.service_name)
    
    # 3. Metrics are automatically initialized upon module import in opentelemetry,
    #    but we would configure the Metrics Exporter here in a production setup.
    #    For now, we rely on default metric providers.
    
    return True
