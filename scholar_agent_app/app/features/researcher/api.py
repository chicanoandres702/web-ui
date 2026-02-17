from fastapi import APIRouter
from app.features.researcher.models import ResearchRequest, ResearchResult
from app.features.researcher.services.research import ResearchService

router = APIRouter()
svc = ResearchService()

@router.post("/search", response_model=ResearchResult)
async def search(req: ResearchRequest):
    res = await svc.perform_research(req.topic)
    return ResearchResult(**res)
