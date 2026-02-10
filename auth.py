import httpx
import os

from urllib.parse import quote_plus, urlencode
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse


router = APIRouter()


@router.get("/login")
async def login(request: Request):

    authorization_url = "https://accounts.google.com/o/oauth2/v2/auth"

    return RedirectResponse(authorization_url, status_code=302)

@router.get("/callback")
async def callback(request: Request, code: str = None, state: str = None):
    return RedirectResponse(url="/", status_code=302)
@router.get("/logout")
async def logout(request: Request):
    return RedirectResponse(url="/", status_code=302)