from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from oauth_flow import initiate_oauth_flow, handle_oauth_callback

router = APIRouter()

@router.get("/auth/login")
async def login(request: Request):
    """Initiates the Google OAuth flow."""
    redirect_url = await initiate_oauth_flow(request)
    return RedirectResponse(redirect_url, status_code=302)

@router.get("/auth/callback")
async def auth_callback(request: Request):
    """Handles the callback from Google OAuth."""
    return await handle_oauth_callback(request)

@router.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")
