
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from career_intelligence.db.session import Base, get_db
from career_intelligence.main import app
from career_intelligence.models import Source

TEST_DATABASE_URL = "sqlite://"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


@pytest.fixture
def client():
    Base.metadata.create_all(bind=engine)

    with TestingSessionLocal() as session:
        session.add(
            Source(
                id=1,
                name="Test Source",
                url="https://example.com/",
                source_type="test",
            )
        )
        session.commit()

    def override_get_db():
        with TestingSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def test_get_source_returns_source(client: TestClient) -> None:
    response = client.get("/sources/1")

    assert response.status_code == 200

    data = response.json()
    assert data["id"] == 1
    assert data["name"] == "Test Source"
    assert data["url"] == "https://example.com/"
    assert data["source_type"] == "test"


def test_get_source_returns_404_when_not_found(client: TestClient) -> None:
    response = client.get("/sources/999999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Source not found."}


def test_delete_source_returns_204(client: TestClient) -> None:
    response = client.delete("/sources/1")

    assert response.status_code == 204
    assert response.content == b""


def test_delete_source_returns_404_when_not_found(
    client: TestClient,
) -> None:
    response = client.delete("/sources/999999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Source not found."}


def test_deleted_source_cannot_be_retrieved(client: TestClient) -> None:
    delete_response = client.delete("/sources/1")

    assert delete_response.status_code == 204

    get_response = client.get("/sources/1")

    assert get_response.status_code == 404
    assert get_response.json() == {"detail": "Source not found."}
