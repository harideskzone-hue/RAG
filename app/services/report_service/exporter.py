import json
from typing import Any


class ReportExporter:
    """
    Exports reports to PDF, CSV, or JSON.
    """
    def export(self, data: dict[str, Any], format_type: str = "json") -> str:
        if format_type == "json":
            return json.dumps(data, indent=2)
            
        # In a real app, generate PDF/CSV here using a configured dependency
        raise NotImplementedError(
            f"Exporting to '{format_type}' requires a configured document generator. "
            "Production fake implementations have been removed."
        )
