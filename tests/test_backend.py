import pytest
from fastapi.testclient import TestClient
from app.backend import create_app

# Fixture to create a TestClient instance for the application
@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient for the FastAPI app.
    """
    app = create_app()
    with TestClient(app) as c:
        yield c

def test_health_check(client: TestClient):
    """
    Test the /health endpoint.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_read_main(client: TestClient):
    """
    Test the root endpoint, which should serve the static index.html file.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers['content-type']
    assert b"<title>Scholar Pro | Command Center</title>" in response.content

# More tests will be added here
