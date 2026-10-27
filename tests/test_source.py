
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



def test_patch_source_updates_only_provided_fields(
    client: TestClient,
) -> None:
    response = client.patch(
        "/sources/1",
        json={"source_type": "tech_community"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["id"] == 1
    assert data["name"] == "Test Source"
    assert data["url"] == "https://example.com/"
    assert data["source_type"] == "tech_community"


def test_patch_source_returns_404_when_not_found(
    client: TestClient,
) -> None:
    response = client.patch(
        "/sources/999999",
        json={"source_type": "tech_community"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Source not found."}


def test_patch_source_rejects_empty_payload(
    client: TestClient,
) -> None:
    response = client.patch("/sources/1", json={})

    assert response.status_code == 422


def test_patch_source_rejects_null_field(
    client: TestClient,
) -> None:
    response = client.patch(
        "/sources/1",
        json={"name": None},
    )

    assert response.status_code == 422



def test_patch_source_returns_409_for_duplicate_name(
    client: TestClient,
) -> None:
    create_response = client.post(
        "/sources",
        json={
            "name": "Another Source",
            "url": "https://another.example.com/",
            "source_type": "test",
        },
    )
    assert create_response.status_code == 201

    response = client.patch(
        "/sources/1",
        json={"name": "Another Source"},
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "A source with this name already exists."
    }

