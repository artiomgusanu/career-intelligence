
from sqlalchemy import select
from sqlalchemy.orm import Session

from career_intelligence.models import Source
from career_intelligence.schemas import SourceUpdate


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


def update_source(
    session: Session,
    source_id: int,
    updates: SourceUpdate,
) -> Source | None:
    source = session.get(Source, source_id)

    if source is None:
        return None

    update_data = updates.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        if field == "url":
            value = str(value)

        setattr(source, field, value)

    session.commit()
    session.refresh(source)

    return source


def delete_source(session: Session, source_id: int) -> bool:
    source = session.get(Source, source_id)

    if source is None:
        return False

    session.delete(source)
    session.commit()
    return True
