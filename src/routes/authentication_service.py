from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app import user_service # type: ignore
from app.exceptions import InvalidCredentialsError
from typing import TYPE_CHECKING
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthenticationService:
    def __init__(self, user_service: user_service.UserService):
        self.user_service = user_service

    async def authenticate_user(self, token: str = Depends(oauth2_scheme)):
        """Authenticates a user based on the provided token."""
        try:
            user = await self.user_service.get_user_by_token(token)
            if not user:
                raise InvalidCredentialsError("Invalid authentication credentials")
            return user
        except InvalidCredentialsError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication failed: {str(e)}",
            )


async def get_authentication_service(user_service: 'user_service.UserService' = Depends(user_service.get_user_service)):
    """Dependency provider for AuthenticationService."""
    return AuthenticationService(user_service=user_service)