from opentelemetry import metrics

meter = metrics.get_meter("vista.system")

# Normally Memory and CPU would be collected automatically by the host metric instrumentation
# opentelemetry-instrumentation-system but we define explicit app-level counters here if needed.

active_workflows = meter.create_up_down_counter(
    "vista.system.active_workflows",
    description="Number of currently active workflows"
)

memory_usage = meter.create_observable_gauge(
    "vista.system.memory_usage",
    description="Application memory usage"
)

cpu_usage = meter.create_observable_gauge(
    "vista.system.cpu_usage",
    description="Application CPU usage"
)
