
from datetime import datetime

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


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


class JobPostingCreate(BaseModel):
    """A normalized job posting for one source, as accepted by the upsert.

    ``source_id`` is not part of the payload: a batch is always upserted for a
    single source. ``first_seen_at`` and ``last_seen_at`` are managed by the
    database. String length limits mirror the database columns so oversize
    values fail validation instead of aborting a batch mid-transaction.

    Optional fields left as ``None`` never erase data already stored for the
    posting (see ``upsert_job_postings``).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    external_id: str = Field(min_length=1, max_length=2048)
    title: str = Field(min_length=1, max_length=300)
    company_name: str = Field(min_length=1, max_length=200)
    job_url: str = Field(min_length=1, max_length=2048)

    company_slug: str | None = Field(default=None, max_length=200)
    application_url: str | None = Field(default=None, max_length=2048)
    employment_type: str | None = Field(default=None, max_length=100)
    categories: str | None = None
    seniority: str | None = None
    location_restrictions: str | None = None
    timezone_restrictions: str | None = None
    salary_min: float | None = Field(default=None, ge=0)
    salary_max: float | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, max_length=10)
    salary_period: str | None = Field(default=None, max_length=30)
    description: str | None = None
    published_at: AwareDatetime | None = None
    expires_at: AwareDatetime | None = None

    # Optional text fields: a blank value ("" or whitespace only) means "not
    # provided" and becomes None, so the upsert never overwrites stored data
    # with an empty string. Required fields and identifiers are not affected.
    @field_validator(
        "company_slug",
        "application_url",
        "employment_type",
        "categories",
        "seniority",
        "location_restrictions",
        "timezone_restrictions",
        "salary_currency",
        "salary_period",
        "description",
    )
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        # Runs after str_strip_whitespace and length validation.
        if value is not None and not value:
            return None
        return value

    @field_validator("salary_currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        return value.upper() if value is not None else None

    @model_validator(mode="after")
    def validate_salary_range(self) -> "JobPostingCreate":
        if (
            self.salary_min is not None
            and self.salary_max is not None
            and self.salary_min > self.salary_max
        ):
            raise ValueError("salary_min cannot be greater than salary_max.")
        return self


class JobPostingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_id: int
    external_id: str
    title: str
    company_name: str
    company_slug: str | None
    job_url: str
    application_url: str | None
    employment_type: str | None
    categories: str | None
    seniority: str | None
    location_restrictions: str | None
    timezone_restrictions: str | None
    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    salary_period: str | None
    description: str | None
    published_at: datetime | None
    expires_at: datetime | None
    first_seen_at: datetime
    last_seen_at: datetime
