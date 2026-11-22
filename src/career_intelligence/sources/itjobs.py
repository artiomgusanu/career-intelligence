"""ITJobs.pt source: record mapper (pure).

Only the mapping step exists for now; the HTTP adapter comes later. The record
shape is taken from the public API documentation (``/api/docs/job``), whose
only example is a ``job/get`` response. It has not been confirmed against real
``job/list`` responses.

Mapped fields:

- ``id`` (positive integer)  -> ``external_id``
- ``title``                  -> ``title``
- ``company.name``           -> ``company_name`` (a ``company`` object is required)
- ``company.slug``           -> ``company_slug``
- ``types[].name``           -> ``employment_type`` (names as given, joined)
- ``body`` (HTML, untrusted) -> ``description`` (stored as given; sanitize on display)

``job_url`` is never produced. The API has no field with the posting's public
URL and the site's URL pattern is not documented, so none is derived from
``id``, ``slug`` or any other field. ``job_url`` is required by
``JobPostingCreate``, so every ITJobs record is rejected at validation
(``job_url: Field required``) until a confirmed source for it exists.

Deliberately not read:

- ``publishedAt`` / ``updatedAt``: no timezone, none is assumed.
- ``salaryMin`` / ``salaryMax`` / ``wage``: currency and period undocumented.
- ``locations`` / ``country``: semantics not confirmed.
- ``contracts`` / ``workModel``: separate dimensions, never mixed into ``types``.
- ``company`` contact data (``email``, ``phone``, ``fax``, ``address``) and
  ``company.url`` / ``logo`` / ``description``, ``companyId``, ``ref``, ``slug``.

Nothing is inferred about a posting's status: no record is marked as expired
or closed. Scalar types (``title``, ``company.name``, ``body``) are left to
``JobPostingCreate``; this module validates structure only.
"""

from collections.abc import Mapping
from typing import Any

from career_intelligence.ingestion.contracts import MappingError, RawPosting


def map_itjobs_posting(raw: RawPosting) -> dict[str, Any]:
    """Normalize one ITJobs record into ``JobPostingCreate`` fields. Pure function.

    Reads a fixed set of keys and never iterates the record, so unknown or
    excluded fields cannot leak into the result. ``raw`` is not modified.
    """
    company = raw.get("company")
    if not isinstance(company, Mapping):
        raise MappingError("company must be an object")

    return {
        "external_id": _external_id(raw.get("id")),
        "title": raw.get("title"),
        "company_name": company.get("name"),
        "company_slug": company.get("slug"),
        "employment_type": _employment_type(raw.get("types")),
        "description": raw.get("body"),
    }


def _external_id(value: object) -> str:
    # bool is a subclass of int and must not pass as an id.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MappingError("id must be a positive integer")
    return str(value)


def _employment_type(types: object) -> str | None:
    if types is None:
        return None
    if not isinstance(types, list):
        raise MappingError("types must be a list")

    names: list[str] = []
    for item in types:
        if not isinstance(item, Mapping):
            raise MappingError("types items must be objects")
        name = item.get("name")
        if not isinstance(name, str):
            raise MappingError("types[].name must be a string")
        name = name.strip()
        if name and name not in names:
            names.append(name)
    return ", ".join(names) or None
