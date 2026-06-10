import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)


def setup_tracing() -> None:
    service_name = os.getenv("OTEL_SERVICE_NAME", "citerag")
    exporter_mode = os.getenv("OTEL_EXPORTER_MODE", "console").lower()

    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": os.getenv("APP_ENV", "dev"),
        }
    )

    provider = TracerProvider(resource=resource)

    if exporter_mode == "otlp":
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)