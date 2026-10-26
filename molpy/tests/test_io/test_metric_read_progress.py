"""apply_request reports how far it read the raw stream (molab's resume cursor)."""

from __future__ import annotations

from dataclasses import dataclass, field

from molpy.io.metric_reader import apply_request


@dataclass
class Progress:
    position: int | None = None


@dataclass
class Request:
    since: int = 0
    stride: int = 1
    limit: int | None = None
    metric_type: str | None = None
    keys: tuple[str, ...] = ()
    progress: Progress | None = field(default_factory=Progress)


def _stream(n_rows: int) -> list[dict]:
    """Rows of two interleaved series, a and b: raw index = 2 * row + series."""
    return [{"k": k, "s": row, "v": row} for row in range(n_rows) for k in ("a", "b")]


def _page(records, request):
    out = []
    for record in apply_request(request, records):
        out.append((record, request.progress.position))
    return out


def test_position_is_the_raw_index_of_each_yielded_record():
    request = Request(keys=("b",))
    got = _page(_stream(3), request)
    assert [position for _, position in got] == [1, 3, 5]


def test_position_after_the_stream_ends_is_its_length():
    request = Request(keys=("a",))
    list(apply_request(request, _stream(3)))
    assert request.progress.position == 6


def test_position_after_a_limit_stops_after_the_last_yielded_record():
    request = Request(keys=("a",), limit=2)
    list(apply_request(request, _stream(5)))
    assert request.progress.position == 3  # yielded raw 0 and 2; resume at 3


def test_position_never_drops_below_since():
    request = Request(since=50)
    assert list(apply_request(request, _stream(3))) == []
    assert request.progress.position == 50


def test_no_limit_reports_since_and_yields_nothing():
    request = Request(since=4, limit=0)
    assert list(apply_request(request, _stream(3))) == []
    assert request.progress.position == 4


def test_paging_by_position_equals_one_full_read():
    records = _stream(20)
    full = list(apply_request(Request(keys=("b",), progress=None), records))
    pages, since = [], 0
    while True:
        request = Request(since=since, keys=("b",), limit=3)
        page = list(apply_request(request, records))
        if not page:
            break
        pages.extend(page)
        since = request.progress.position
    assert pages == full


def test_a_request_without_progress_still_reads():
    @dataclass
    class Bare:
        since: int = 0
        stride: int = 1
        limit: int | None = None
        metric_type: str | None = None
        keys: tuple[str, ...] = ()

    assert len(list(apply_request(Bare(), _stream(2)))) == 4
