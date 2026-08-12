from datetime import datetime, timedelta


class ClipSelector:
    """
    Selects optimal video segments based on metadata/evidence timestamps to minimize VLM latency and cost.
    """
    def select_clip_window(self, timestamp: datetime, padding_seconds: int = 5) -> tuple[datetime, datetime]:
        """
        Creates a time window around a specific event timestamp.
        """
        start = timestamp - timedelta(seconds=padding_seconds)
        end = timestamp + timedelta(seconds=padding_seconds)
        return start, end
