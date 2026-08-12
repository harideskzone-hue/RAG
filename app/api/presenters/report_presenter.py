from app.api.schemas.response import ReportResponse


class ReportPresenter:
    @staticmethod
    def present_job_created(job_id: str) -> ReportResponse:
        return ReportResponse(
            job_id=job_id,
            status="pending",
            message="Report generation job created successfully."
        )
