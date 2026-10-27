
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


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


class SourceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=150)
    url: HttpUrl | None = None
    source_type: str | None = Field(default=None, min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_update(self) -> "SourceUpdate":
        if not self.model_fields_set:
            raise ValueError("At least one field must be provided.")

        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be null.")

        return self
