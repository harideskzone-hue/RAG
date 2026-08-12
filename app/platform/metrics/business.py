from opentelemetry import metrics

meter = metrics.get_meter("vista.business")

investigations_started = meter.create_counter(
    "vista.business.investigations_started",
    description="Number of investigations started"
)

reports_generated = meter.create_counter(
    "vista.business.reports_generated",
    description="Number of reports generated"
)

person_searches = meter.create_counter(
    "vista.business.person_searches",
    description="Number of person searches executed"
)
