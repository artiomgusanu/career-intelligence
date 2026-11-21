"""Generic page-number pagination for source adapters."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from career_intelligence.ingestion.contracts import IngestionError


@dataclass(frozen=True)
class Page[T]:
    items: Sequence[T]
    has_next: bool


class IncompletePaginationError(IngestionError):
    """Raised when pagination cannot be completed.

    Returning the items gathered so far would make a partial collection look
    complete, so pagination never stops silently.
    """


class PaginationLimitError(IncompletePaginationError):
    """Raised when ``max_pages`` is reached while the source still reports more pages."""


def paginate[T](
    fetch_page: Callable[[int], Page[T]],
    *,
    first_page: int = 1,
    max_pages: int = 1000,
) -> Iterator[T]:
    """Yield items page by page until the source reports no further pages.

    An empty page ends pagination only if the source also reports no further
    pages; an empty page with ``has_next`` set raises
    ``IncompletePaginationError``. Pages are fetched lazily.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1.")

    page_number = first_page
    for _ in range(max_pages):
        page = fetch_page(page_number)

        if not page.items:
            if page.has_next:
                raise IncompletePaginationError(
                    f"Page {page_number} was empty while the source reports more pages."
                )
            return

        yield from page.items
        if not page.has_next:
            return
        page_number += 1

    raise PaginationLimitError(
        f"Stopped after {max_pages} pages while the source still reports more pages."
    )
