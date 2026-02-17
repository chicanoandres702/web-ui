from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

class AuthStatus(BaseModel):
    is_logged_in: bool
    user: dict | None = None

@router.get("/status", response_model=AuthStatus)
async def auth_status(request: Request):
    user = request.session.get("user")
    return AuthStatus(is_logged_in=bool(user), user=user)

@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"status": "logged_out"}
