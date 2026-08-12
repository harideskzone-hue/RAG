import uuid

from fastapi import APIRouter, Depends

from app.api.dependencies.security import get_current_user, require_permission
from app.api.dependencies.supervisor import get_supervisor
from app.api.presenters.report_presenter import ReportPresenter
from app.api.schemas.request import ReportRequest
from app.api.schemas.response import ReportResponse
from app.background.service import BackgroundTaskService
from app.graph.supervisor.supervisor import Supervisor
from app.schemas.context import UserContext, VistaContext

router = APIRouter(prefix="/reports", tags=["reports"])

def get_background_task_service() -> BackgroundTaskService:
    return BackgroundTaskService()

async def run_report_job(supervisor: Supervisor, context: VistaContext):
    await supervisor.run(context)
    # The output is saved somewhere (e.g., S3), in a real setup we'd mark DB job as finished

@router.post("", response_model=ReportResponse, dependencies=[Depends(require_permission("write:report"))])
async def generate_report(
    payload: ReportRequest,
    supervisor: Supervisor = Depends(get_supervisor),
    current_user: dict = Depends(get_current_user),
    bg_service: BackgroundTaskService = Depends(get_background_task_service)
):
    job_id = str(uuid.uuid4())
    
    context = VistaContext(
        user=UserContext(
            user_id=current_user.get("sub", "unknown"),
            role=current_user.get("role", "viewer")
        ),
        conversation_id=job_id,
        current_query=payload.query
    )
    
    # Submit async job
    await bg_service.submit(job_id, run_report_job, supervisor, context)
    
    return ReportPresenter.present_job_created(job_id)
