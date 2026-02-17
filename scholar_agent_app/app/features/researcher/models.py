from pydantic import BaseModel
from typing import List, Optional

class ResearchRequest(BaseModel):
    topic: str
    depth: str = "standard"

class ResearchResult(BaseModel):
    summary: str
    sources: List[str]
