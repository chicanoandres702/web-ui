class AuthenticationError(Exception):
    """Base class for authentication exceptions."""
    pass

class InvalidCredentialsError(AuthenticationError):
    """Raised when provided credentials are invalid."""
    pass