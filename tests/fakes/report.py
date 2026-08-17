import json
from typing import Any
from app.services.report_service.exporter import ReportExporter

class FakeReportExporter(ReportExporter):
    """
    Fake report exporter for tests.
    """
    def export(self, data: dict[str, Any], format_type: str = "json") -> str:
        if format_type == "json":
            return json.dumps(data, indent=2)
            
        return f"Mock {format_type.upper()} report."
