import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import pytest
from bk_data.calendar import (
    XNYS,
    Date,
    is_session,
    next_session,
    previous_session,
    sessions,
    sessions_window,
)
from bk_data.exceptions import CalendarError, CalendarTypeError, CalendarValueError
from pandas import Timestamp

# Every public function, called with a single date, so input validation can be
# tested across all of them at once.
ALL_FUNCTIONS: list[Callable[[Any], object]] = [
    is_session,
    next_session,
    previous_session,
    lambda d: sessions(d, d),
    lambda d: sessions_window(d, 1),
]
ALL_FUNCTION_IDS = [
    "is_session",
    "next_session",
    "previous_session",
    "sessions",
    "sessions_window",
]

all_functions = pytest.mark.parametrize("func", ALL_FUNCTIONS, ids=ALL_FUNCTION_IDS)


def test_trading_days() -> None:
    trading_days = sessions(date(2023, 1, 1), date(2024, 1, 1))
    assert len(trading_days) == 250


@pytest.mark.parametrize(
    "start,days,expected",
    [
        (
            date(2026, 1, 2),
            7,
            [
                Timestamp("2026-01-02"),
                Timestamp("2026-01-05"),
                Timestamp("2026-01-06"),
                Timestamp("2026-01-07"),
                Timestamp("2026-01-08"),
                Timestamp("2026-01-09"),
                Timestamp("2026-01-12"),
            ],
        ),
        (
            date(2026, 1, 5),
            -4,
            [
                Timestamp("2025-12-30"),
                Timestamp("2025-12-31"),
                Timestamp("2026-01-02"),
                Timestamp("2026-01-05"),
            ],
        ),
    ],
)
def test_sessions_window(start: Date, days: int, expected: list[Timestamp]) -> None:
    window = sessions_window(start, days)
    assert list(window) == expected


@pytest.mark.parametrize(
    "first_session,expected",
    [
        (date(2025, 12, 31), Timestamp("2026-01-02")),
        (Timestamp(2026, 1, 2), Timestamp("2026-01-05")),
        ("2026-01-05", Timestamp("2026-01-06")),
    ],
)
def test_next_session(first_session: Date, expected: Timestamp) -> None:
    next_ = next_session(first_session)
    assert next_ == expected


@pytest.mark.parametrize(
    "first_session,expected",
    [
        (date(2026, 1, 2), Timestamp("2025-12-31")),
        (Timestamp(2026, 1, 5), Timestamp("2026-01-02")),
        ("2026-01-06", Timestamp("2026-01-05")),
    ],
)
def test_previous_session(first_session: Date, expected: Timestamp) -> None:
    prev_ = previous_session(first_session)
    assert prev_ == expected


@pytest.mark.parametrize(
    "date,expected",
    [
        (date(2026, 1, 1), False),
        ("2026-01-02", True),
        (datetime(2026, 1, 3), False),
        ("2025-01-09", False),
        ("2023-06-19", False),
        ("2025-11-28", True),
        ("2024-03-29", False),
    ],
)
def test_is_session(date: Date, expected: bool) -> None:
    assert expected == is_session(date)


@all_functions
@pytest.mark.parametrize("val", [20260102, 2026.0, None, ["2026-01-02"]])
def test_wrong_type(func: Callable[[Any], object], val: Any) -> None:
    with pytest.raises(CalendarTypeError, match="Invalid date type"):
        func(val)


@all_functions
@pytest.mark.parametrize(
    "val",
    [
        datetime(2026, 1, 2, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=timezone(timedelta(hours=-5))),
        Timestamp("2026-01-02", tz="America/New_York"),
    ],
)
def test_timezone_aware(func: Callable[[Any], object], val: Date) -> None:
    with pytest.raises(CalendarValueError, match="timezone"):
        func(val)


@all_functions
@pytest.mark.parametrize(
    "val", ["01/02/2026", "2 Jan 2026", "2026/01/02", "2026-01-02T15:30", ""]
)
def test_non_iso_string(func: Callable[[Any], object], val: str) -> None:
    with pytest.raises(CalendarValueError, match="iso date format"):
        func(val)


@pytest.mark.parametrize(
    "func",
    [next_session, previous_session, lambda d: sessions_window(d, 3)],
    ids=["next_session", "previous_session", "sessions_window"],
)
@pytest.mark.parametrize(
    "val",
    [
        "2026-01-03",  # Saturday
        date(2026, 1, 4),  # Sunday
        "2026-01-01",  # New Year's Day
        "2024-03-29",  # Good Friday
    ],
)
def test_requires_session(func: Callable[[Any], object], val: Date) -> None:
    with pytest.raises(CalendarValueError, match="is not a session"):
        func(val)


@all_functions
@pytest.mark.parametrize(
    "val", [datetime(2026, 1, 2, 15, 30), Timestamp("2026-01-02 09:30")]
)
def test_datetime_with_time(func: Callable[[Any], object], val: Date) -> None:
    with pytest.raises(CalendarValueError):
        func(val)


@all_functions
@pytest.mark.parametrize("val", ["1990-01-02", "2099-01-02"])
def test_out_of_range(func: Callable[[Any], object], val: str) -> None:
    with pytest.raises(CalendarValueError):
        func(val)


@pytest.mark.parametrize(
    "val,exc,message",
    [
        (20260102, CalendarTypeError, "Invalid date type int, please use date, "
         "datetime, str or pandas Timestamp."),
        ("01/02/2026", CalendarValueError,
         "Date string '01/02/2026' is not in iso date format (YYYY-MM-DD)."),
        (Timestamp("2026-01-02", tz="UTC"), CalendarValueError,
         "2026-01-02 00:00:00+00:00 should not have a timezone."),
        (datetime(2026, 1, 2, 15, 30), CalendarValueError,
         "2026-01-02 15:30:00 must be at midnight."),
        (Timestamp("2026-01-02 00:00:00.000000001"), CalendarValueError,
         "2026-01-02 00:00:00.000000001 must be at midnight."),
        (date(1990, 1, 2), CalendarValueError,
         "1990-01-02 should not be before calendar start 2000-01-03."),
        ("2026-01-03", CalendarValueError, "2026-01-03 is not a session."),
    ],
)  # fmt: skip
def test_full_error_message(val: Any, exc: type[Exception], message: str) -> None:
    with pytest.raises(exc, match=f"^{re.escape(message)}$"):
        next_session(val)


@pytest.mark.parametrize("val", [date(2099, 1, 2), datetime(2099, 1, 2), "2099-01-02"])
def test_after_calendar_end_message(val: Date) -> None:
    message = f"2099-01-02 should not be after calendar end {XNYS.last_session.date()}."
    with pytest.raises(CalendarValueError, match=f"^{re.escape(message)}$"):
        is_session(val)


def test_sessions_end_before_start() -> None:
    message = "end 2026-01-05 is before start 2026-01-10."
    with pytest.raises(CalendarValueError, match=f"^{re.escape(message)}$"):
        sessions("2026-01-10", "2026-01-05")


def test_sessions_same_day() -> None:
    assert list(sessions("2026-01-02", "2026-01-02")) == [Timestamp("2026-01-02")]


@pytest.mark.parametrize("exc", [CalendarError, CalendarTypeError, CalendarValueError])
def test_calendar_errors_are_exceptions(exc: type[BaseException]) -> None:
    assert issubclass(exc, Exception)


def test_calendar_errors_share_base() -> None:
    assert issubclass(CalendarTypeError, CalendarError)
    assert issubclass(CalendarValueError, CalendarError)
