
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from career_intelligence.db.repositories import (
    create_source,
    get_source_by_id,
    list_sources,
)
from career_intelligence.db.session import Base


def test_source_repository_persists_data() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        source = create_source(
            session,
            name="Example Source",
            url="https://example.com",
            source_type="website",
        )

        retrieved = get_source_by_id(session, source.id)
        sources = list_sources(session)

        assert retrieved is not None
        assert retrieved.name == "Example Source"
        assert retrieved.url == "https://example.com"
        assert retrieved.source_type == "website"
        assert len(sources) == 1

    engine.dispose()
