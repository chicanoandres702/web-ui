import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.backend import create_app

@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient for the FastAPI app.
    """
    app = create_app()
    with TestClient(app) as c:
        yield c

def test_login_page(client: TestClient):
    """
    Test that the login page is served correctly.
    """
    response = client.get("/login_page")
    assert response.status_code == 200
    assert b"Login with Google" in response.content

def test_logout_page(client: TestClient):
    """
    Test that the logout page is served correctly.
    """
    response = client.get("/logout_page")
    assert response.status_code == 200
    assert b"You have been successfully logged out." in response.content

@patch('app.auth.Flow')
def test_auth_login(mock_flow, client: TestClient):
    """
    Test the /auth/login endpoint.
    """
    # Mock the Flow instance and its methods
    mock_flow_instance = MagicMock()
    mock_flow_instance.authorization_url.return_value = ("https://fake-auth-url.com", "fake-state")
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance
    mock_flow.from_client_config.return_value = mock_flow_instance

    response = client.get("/auth/login")
    
    # We expect a redirect to the authorization URL
    assert response.status_code == 307 # Or 302, TestClient uses 307 for redirects
    assert response.headers["location"] == "https://fake-auth-url.com"
    # Check that the state is stored in the session
    assert client.cookies.get("session") is not None

@patch('app.auth.Flow')
def test_auth_callback(mock_flow, client: TestClient):
    """
    Test the /auth/callback endpoint.
    """
    # Mock the Flow instance and its methods
    mock_flow_instance = MagicMock()
    mock_creds = MagicMock()
    mock_creds.token = "fake-token"
    mock_creds.refresh_token = "fake-refresh-token"
    mock_creds.client_id = "fake-client-id"
    mock_creds.scopes = ["scope1", "scope2"]
    mock_flow_instance.credentials = mock_creds
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance
    mock_flow.from_client_config.return_value = mock_flow_instance

    # Set a fake state in the session, as the login endpoint would do
    with client.session_transaction() as sess:
        sess["state"] = "fake-state"
    
    response = client.get("/auth/callback?state=fake-state&code=fake-code")
    
    # We expect a redirect to the root
    assert response.status_code == 307
    assert response.headers["location"] == "/"
    
    # Check that the credentials are in the session after the redirect is handled
    with client.session_transaction() as sess:
        assert "google_creds" in sess
        assert sess["google_creds"]["token"] == "fake-token"

def test_auth_status_logged_out(client: TestClient):
    """
    Test the /auth/status endpoint when logged out.
    """
    response = client.get("/auth/status")
    assert response.status_code == 200
    assert response.json() == {"is_logged_in": False}

def test_auth_status_logged_in(client: TestClient):
    """
    Test the /auth/status endpoint when logged in.
    """
    # Manually set the session credentials
    with client.session_transaction() as sess:
        sess["google_creds"] = {"token": "fake-token"}
        
    response = client.get("/auth/status")
    assert response.status_code == 200
    assert response.json() == {"is_logged_in": True}

def test_auth_logout(client: TestClient):
    """
    Test the /auth/logout endpoint.
    """
    # First, log in
    with client.session_transaction() as sess:
        sess["google_creds"] = {"token": "fake-token"}
    
    # Then, log out
    response = client.get("/auth/logout")
    assert response.status_code == 307
    assert response.headers["location"] == "/"
    
    # Check that the credentials are gone from the session
    with client.session_transaction() as sess:
        assert "google_creds" not in sess
