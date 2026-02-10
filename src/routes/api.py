# src/routes/api.py

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/hello")
async def hello_world():
    """
    A simple hello world endpoint.
    """
    return JSONResponse({"message": "Hello, world!"})