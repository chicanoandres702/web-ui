import os
from urllib.parse import quote_plus, urlencode
import httpx

from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse
from app.config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET


async def initiate_oauth_flow(request: Request) -> str:
    """Constructs the Google OAuth authorization URL."""

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google OAuth not configured.")

    redirect_uri = "http://localhost:8000/callback"
    state = os.urandom(16).hex()
    request.session["state"] = state

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": "challenge",
        "code_challenge_method": "S256"
    }

    authorization_url = "https://accounts.google.com/o/oauth2/v2/auth" + "?" + urlencode(params, quote_via=quote_plus)
    return authorization_url


async def handle_oauth_callback(request: Request):
    """Handles the callback from Google OAuth."""
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if state != request.session.get("state"):
        raise HTTPException(status_code=400, detail="Invalid state")

    token_url = "https://oauth2.googleapis.com/token"
    redirect_uri = "http://localhost:8000/callback"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "code_verifier": "verifier"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=data)
        response.raise_for_status()
        token_data = response.json()

    # Use the token to fetch user info
    user_info_url = "https://openidconnect.googleapis.com/v1/userinfo"
    headers = {"Authorization": f"Bearer {token_data['access_token']}"}

    async with httpx.AsyncClient() as client:
        user_response = await client.get(user_info_url, headers=headers)
        user_response.raise_for_status()
        user_data = user_response.json()

    # Store user info in session
    request.session["user"] = user_data
    request.session["google_creds"] = {
        "token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "token_uri": token_url,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "scopes": ["openid", "email", "profile"]
    }
    return RedirectResponse(url="/", status_code=302)