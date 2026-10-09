
from sqlalchemy import select
from sqlalchemy.orm import Session

from career_intelligence.models import Source


def create_source(
    session: Session,
    *,
    name: str,
    url: str,
    source_type: str,
) -> Source:
    source = Source(
        name=name,
        url=url,
        source_type=source_type,
    )

    session.add(source)
    session.commit()
    session.refresh(source)

    return source


def get_source_by_id(
    session: Session,
    source_id: int,
) -> Source | None:
    return session.get(Source, source_id)


def list_sources(session: Session) -> list[Source]:
    statement = select(Source).order_by(Source.id)
    return list(session.scalars(statement).all())
