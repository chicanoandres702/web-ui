from fastapi import APIRouter, Depends, HTTPException, Request
from src.agent.browser_use.components.knowledge_base import KnowledgeBase
from typing import Dict, Any


def create_knowledge_base_router() -> APIRouter:
    router = APIRouter()

    async def get_knowledge_base(request: Request) -> KnowledgeBase:
        """Dependency to retrieve the KnowledgeBase instance from the application state."""
        return request.app.state.knowledge_base

    @router.post("/knowledge/export")
    async def export_knowledge(filepath: str, knowledge_base: KnowledgeBase = Depends(get_knowledge_base)) -> Dict[str, Any]:
        """Exports the knowledge base to a JSON file."""
        if knowledge_base.export_knowledge(filepath):
            return {"message": f"Knowledge base successfully exported to '{filepath}'."}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to export knowledge base to '{filepath}'.")

    @router.post("/knowledge/{key}")
    async def store_knowledge(key: str, knowledge: dict, knowledge_base: KnowledgeBase = Depends(get_knowledge_base)) -> Dict[str, Any]:
        """Stores knowledge in the knowledge base."""
        if knowledge_base.store_knowledge(key, knowledge):
            return {"message": f"Knowledge stored successfully under key '{key}'."}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to store knowledge under key '{key}'.")

    @router.get("/knowledge/{key}")
    async def retrieve_knowledge(key: str, knowledge_base: KnowledgeBase = Depends(get_knowledge_base)) -> Dict[str, Any]:
        
        """Retrieves knowledge from the knowledge base."""
        knowledge = knowledge_base.retrieve_knowledge(key)
        if knowledge is not None:
            return knowledge

        else:
            raise HTTPException(status_code=404, detail=f"Knowledge not found for key '{key}'.")

    @router.get("/knowledge/import")
    async def import_knowledge(filepath: str, knowledge_base: KnowledgeBase = Depends(get_knowledge_base)) -> Dict[str, Any]:
        """Retrieves all knowledge from the knowledge base."""       
        if knowledge_base.import_knowledge(filepath):
            return {"message": f"Knowledge base successfully imported from '{filepath}'."}
        else:

            raise HTTPException(status_code=500, detail=f"Failed to import knowledge base from '{filepath}'.")
    
    @router.get("/knowledge")
    async def get_all_knowledge(knowledge_base: KnowledgeBase = Depends(get_knowledge_base)) -> Dict[str, Any]:
        """Retrieves all knowledge from the knowledge base."""
        return knowledge_base.storage

    return router