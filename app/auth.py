from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
import os

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Provided Credentials (Fallback)
DEFAULT_CLIENT_ID = "566039560183-8iajns5sv7799tn4k2j6h9rsht19bjkp.apps.googleusercontent.com"
DEFAULT_CLIENT_SECRET = "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"

# Updated Requested Scopes to include userinfo.profile
SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'openid',
    'https://www.googleapis.com/auth/generative-language.peruserquota'
]

@router.get("/login_page")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/logout_page")
async def logout_page(request: Request):
    return templates.TemplateResponse("logout.html", {"request": request})

@router.get("/auth/login")
async def login(request: Request):
    try:
        from google_auth_oauthlib.flow import Flow
        redirect_uri = str(request.url_for('auth_callback'))

        if os.path.exists("client_secret.json"):
            flow = Flow.from_client_secrets_file(
                "client_secret.json",
                scopes=SCOPES,
                redirect_uri=redirect_uri
            )
        else:
            client_id = os.getenv("GOOGLE_CLIENT_ID", DEFAULT_CLIENT_ID)
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET", DEFAULT_CLIENT_SECRET)
            
            client_config = {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
            
            if not client_id or not client_secret:
                return HTMLResponse("Error: GOOGLE_CLIENT_ID/SECRET not configured.")

            flow = Flow.from_client_config(
                client_config,
                scopes=SCOPES,
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
        state = request.session.get('state')
        redirect_uri = str(request.url_for('auth_callback'))

        if os.path.exists("client_secret.json"):
            flow = Flow.from_client_secrets_file(
                "client_secret.json",
                scopes=SCOPES,
                state=state,
                redirect_uri=redirect_uri
            )
        else:
            client_id = os.getenv("GOOGLE_CLIENT_ID", DEFAULT_CLIENT_ID)
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET", DEFAULT_CLIENT_SECRET)
            
            client_config = {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
            flow = Flow.from_client_config(
                client_config,
                scopes=SCOPES,
                state=state,
                redirect_uri=redirect_uri
            )
            
        flow.fetch_token(authorization_response=str(request.url))
        creds = flow.credentials
        request.session['google_creds'] = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'client_id': creds.client_id,
            'scopes': creds.scopes
        }
        return RedirectResponse("/")
    except Exception as e:
        return HTMLResponse(f"Auth Error: {e}")

@router.get("/auth/status")
async def auth_status(request: Request):
    return {"is_logged_in": "google_creds" in request.session}

@router.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")
