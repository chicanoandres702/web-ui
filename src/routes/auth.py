import os
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.config import WEBSOCKET_TOKEN

router = APIRouter()


def get_websocket_token():
    """
    Returns the WebSocket token.  In a real application, this might be fetched
    from a database or configuration file.
    """
    return WEBSOCKET_TOKEN


def verify_websocket_token(token: str = Depends(get_websocket_token)):
    """
    Dependency to verify the WebSocket token.
    """
    async def inner(websocket: WebSocket):
        query_params = websocket.query_params
        client_token = query_params.get("token")
        if not client_token or client_token != token:
            raise HTTPException(
                status_code=401, detail="Invalid WebSocket token"
            )
    return inner


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
        state = request.session.get('state')
        scopes = ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/userinfo.email', 'openid', 'https://www.googleapis.com/auth/generative-language.peruserquota']
        redirect_uri = str(request.url_for('auth_callback'))

        if os.path.exists("client_secret.json"):
            flow = Flow.from_client_secrets_file(
                "client_secret.json",
                scopes=scopes,
                state=state,
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
            flow = Flow.from_client_config(
                client_config,
                scopes=scopes,
                state=state,
                redirect_uri=redirect_uri
            )
            
        flow.fetch_token(authorization_response=str(request.url))
        creds = flow.credentials
        request.session['google_creds'] = {'token': creds.token, 'refresh_token': creds.refresh_token, 'token_uri': creds.token_uri, 'client_id': creds.client_id, 'client_secret': creds.client_secret, 'scopes': creds.scopes}
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

@router.get("/metrics")
async def metrics():
    """Endpoint to expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
