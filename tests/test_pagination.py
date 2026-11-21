import pytest

from career_intelligence.ingestion.contracts import IngestionError
from career_intelligence.ingestion.pagination import (
    IncompletePaginationError,
    Page,
    PaginationLimitError,
    paginate,
)


class FakeSource:
    def __init__(self, pages: dict[int, Page[int]]) -> None:
        self.pages = pages
        self.requested: list[int] = []

    def __call__(self, page_number: int) -> Page[int]:
        self.requested.append(page_number)
        return self.pages[page_number]


def test_yields_items_across_pages_until_has_next_is_false() -> None:
    source = FakeSource(
        {1: Page([1, 2], has_next=True), 2: Page([3, 4], has_next=True), 3: Page([5], False)}
    )

    assert list(paginate(source)) == [1, 2, 3, 4, 5]
    assert source.requested == [1, 2, 3]


def test_empty_page_claiming_more_pages_raises_instead_of_truncating() -> None:
    source = FakeSource(
        {1: Page([1, 2], has_next=True), 2: Page([], has_next=True), 3: Page([3], False)}
    )
    collected: list[int] = []

    with pytest.raises(IncompletePaginationError, match="Page 2 was empty"):
        for item in paginate(source):
            # Explicit loop: keeps the items yielded before the generator raises.
            collected.append(item)  # noqa: PERF402

    assert collected == [1, 2]
    assert source.requested == [1, 2]  # page 3 is never requested


def test_empty_final_page_ends_pagination_normally() -> None:
    source = FakeSource({1: Page([1], has_next=True), 2: Page([], has_next=False)})

    assert list(paginate(source)) == [1]


def test_empty_source_yields_nothing() -> None:
    assert list(paginate(FakeSource({1: Page([], has_next=False)}))) == []


def test_pagination_failures_are_ingestion_errors() -> None:
    source = FakeSource({n: Page([n], has_next=True) for n in range(1, 5)})

    with pytest.raises(IngestionError) as exc_info:
        list(paginate(source, max_pages=2))

    assert isinstance(exc_info.value, IncompletePaginationError)


def test_respects_first_page() -> None:
    source = FakeSource({0: Page(["a"], has_next=True), 1: Page(["b"], has_next=False)})

    assert list(paginate(source, first_page=0)) == ["a", "b"]


def test_raises_instead_of_truncating_when_max_pages_reached() -> None:
    source = FakeSource({n: Page([n], has_next=True) for n in range(1, 10)})
    collected: list[int] = []

    with pytest.raises(PaginationLimitError):
        for item in paginate(source, max_pages=3):
            # Explicit loop: keeps the items yielded before the generator raises.
            collected.append(item)  # noqa: PERF402

    assert collected == [1, 2, 3]
    assert source.requested == [1, 2, 3]


def test_exactly_max_pages_with_no_more_is_not_an_error() -> None:
    source = FakeSource({1: Page([1], has_next=True), 2: Page([2], has_next=False)})

    assert list(paginate(source, max_pages=2)) == [1, 2]


def test_pages_are_fetched_lazily() -> None:
    source = FakeSource({1: Page([1, 2], has_next=True), 2: Page([3], has_next=False)})

    iterator = paginate(source)
    assert next(iterator) == 1

    assert source.requested == [1]


def test_max_pages_must_be_positive() -> None:
    with pytest.raises(ValueError):
        list(paginate(FakeSource({}), max_pages=0))
