from opentelemetry import metrics

meter = metrics.get_meter("vista.infrastructure")

postgres_latency = meter.create_histogram(
    "vista.infrastructure.postgres_latency",
    description="Latency of PostgreSQL queries",
    unit="ms"
)

redis_latency = meter.create_histogram(
    "vista.infrastructure.redis_latency",
    description="Latency of Redis operations",
    unit="ms"
)

milvus_latency = meter.create_histogram(
    "vista.infrastructure.milvus_latency",
    description="Latency of Milvus vector queries",
    unit="ms"
)

s3_latency = meter.create_histogram(
    "vista.infrastructure.s3_latency",
    description="Latency of S3 object storage operations",
    unit="ms"
)
