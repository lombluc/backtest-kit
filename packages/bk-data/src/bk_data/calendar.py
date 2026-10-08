from datetime import date, datetime

import exchange_calendars as xcals
from pandas import DatetimeIndex, Timestamp

from bk_data.exceptions import CalendarTypeError, CalendarValueError

CALENDAR_START = "2000-01-01"
XNYS = xcals.get_calendar("XNYS", start=CALENDAR_START)


Date = date | datetime | str | Timestamp


def check_date(date_: Date) -> Timestamp:
    """Validate a date and return it as a tz-naive midnight Timestamp."""
    if not isinstance(date_, Date):
        raise CalendarTypeError(
            f"Invalid date type {type(date_).__name__}, please use date, datetime, "
            "str or pandas Timestamp."
        )

    if isinstance(date_, str):
        try:
            date.fromisoformat(date_)
        except ValueError:
            raise CalendarValueError(
                f"Date string {date_!r} is not in iso date format (YYYY-MM-DD)."
            ) from None

    ts = Timestamp(date_)

    if ts.tz is not None:
        raise CalendarValueError(f"{date_} should not have a timezone.")
    # Compare against normalize() rather than time(), which drops nanoseconds.
    if ts != ts.normalize():
        raise CalendarValueError(f"{date_} must be at midnight.")
    if ts > XNYS.last_session:
        raise CalendarValueError(
            f"{ts.date()} should not be after calendar end {XNYS.last_session.date()}."
        )
    if ts < XNYS.first_session:
        raise CalendarValueError(
            f"{ts.date()} should not be before calendar start "
            f"{XNYS.first_session.date()}."
        )

    return ts


def _require_session(ts: Timestamp) -> None:
    if not XNYS.is_session(ts):
        raise CalendarValueError(f"{ts.date()} is not a session.")


def sessions(start: Date, end: Date) -> DatetimeIndex:
    start_ts = check_date(start)
    end_ts = check_date(end)
    if end_ts < start_ts:
        raise CalendarValueError(
            f"end {end_ts.date()} is before start {start_ts.date()}."
        )
    return DatetimeIndex(XNYS.sessions_in_range(start_ts, end_ts))


def sessions_window(start: Date, days: int) -> DatetimeIndex:
    ts = check_date(start)
    _require_session(ts)
    return DatetimeIndex(XNYS.sessions_window(ts, days))


def next_session(date_: Date) -> Timestamp:
    ts = check_date(date_)
    _require_session(ts)
    return Timestamp(XNYS.next_session(ts))


def previous_session(date_: Date) -> Timestamp:
    ts = check_date(date_)
    _require_session(ts)
    return Timestamp(XNYS.previous_session(ts))


def is_session(date_: Date) -> bool:
    ts = check_date(date_)
    return bool(XNYS.is_session(ts))
