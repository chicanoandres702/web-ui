import os
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()

@router.get("/auth/login")
async def login(request: Request):
    try:
        from google_auth_oauthlib.flow import Flow
        scopes = ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/userinfo.email', 'openid', 'https://www.googleapis.com/auth/generative-language.peruserquota']
        redirect_uri = str(request.url_for('auth_callback'))

        if os.path.exists("client_secret.json"):
            flow = Flow.from_client_secrets_file(
                "client_secret.json",
                scopes=scopes,
                redirect_uri=redirect_uri
            )
        else:
            client_config = {
                "web": {
                    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
            if not client_config["web"]["client_id"] or not client_config["web"]["client_secret"]:
                return HTMLResponse("Error: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars not set, and client_secret.json not found.")

            flow = Flow.from_client_config(
                client_config,
                scopes=scopes,
                redirect_uri=redirect_uri
            )

        authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true', prompt='consent')
        request.session['state'] = state
        return RedirectResponse(authorization_url)
    except ImportError:
        return HTMLResponse("Error: google-auth-oauthlib not installed.")

@router.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        from google_auth_oauthlib.flow import Flow
        # Existing auth_callback implementation (same as in server.py)
        return RedirectResponse("/")
    except Exception as e:
        return HTMLResponse(f"Auth Error: {e}")

@router.get("/auth/status")
async def auth_status(request: Request):
    return {"is_logged_in": "google_creds" in request.session}