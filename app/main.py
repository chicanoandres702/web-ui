import os
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

router = APIRouter()

@router.get("/health")
async def health_check():    
   return {"status": "ok"}

# def get_index_path():
#    """Helper to get the path to the index.html file."""
#    static_dir = os.getenv("STATIC_DIR", "static")
#    return os.path.join("./static/", "index.html")

# @router.get("/", include_in_schema=False)
# async def serve_index():
#    """Serves the index.html file from the static directory."""
#    return FileResponse(get_index_path())