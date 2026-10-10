
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    url: HttpUrl
    source_type: str = Field(min_length=1, max_length=50)


class SourceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    url: str
    source_type: str
    created_at: datetime
