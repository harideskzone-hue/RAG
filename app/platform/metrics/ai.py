from opentelemetry import metrics

meter = metrics.get_meter("vista.ai")

prompt_tokens = meter.create_counter(
    "vista.ai.prompt_tokens",
    description="Number of prompt tokens sent to LLMs/VLMs"
)

completion_tokens = meter.create_counter(
    "vista.ai.completion_tokens",
    description="Number of completion tokens received"
)

vlm_calls = meter.create_counter(
    "vista.ai.vlm_calls",
    description="Number of Vision Language Model calls"
)

confidence_average = meter.create_histogram(
    "vista.ai.confidence",
    description="Distribution of confidence scores"
)
