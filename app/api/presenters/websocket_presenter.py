import json


class WebSocketPresenter:
    @staticmethod
    def present_progress(stage: str, progress: int) -> str:
        return json.dumps({
            "type": "progress",
            "stage": stage,
            "progress": progress
        })

    @staticmethod
    def present_token(text: str) -> str:
        return json.dumps({
            "type": "token",
            "text": text
        })

    @staticmethod
    def present_completed(result: dict) -> str:
        return json.dumps({
            "type": "completed",
            "result": result
        })
