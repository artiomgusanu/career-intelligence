
from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from career_intelligence.db.repositories import create_source, list_sources
from career_intelligence.db.session import get_db
from career_intelligence.schemas import SourceCreate, SourceResponse

app = FastAPI(
    title="Career Intelligence",
    description="An intelligence platform for career and market research.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/sources",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_source_endpoint(
    payload: SourceCreate,
    db: Session = Depends(get_db),
) -> SourceResponse:
    try:
        source = create_source(
            db,
            name=payload.name,
            url=str(payload.url),
            source_type=payload.source_type,
        )
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A source with this name already exists.",
        ) from exc

    return SourceResponse.model_validate(source)


@app.get("/sources", response_model=list[SourceResponse])
def get_sources(
    db: Session = Depends(get_db),
) -> list[SourceResponse]:
    return list_sources(db)
