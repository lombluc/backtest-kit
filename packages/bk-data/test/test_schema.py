from datetime import date, datetime, timedelta, timezone
import re

import pytest
from bk_data.exceptions import BarValidationError, DateParseError
from bk_data.schema import validate_bars
from pandas import DataFrame, Timestamp, concat

df0 = DataFrame(
    {
        "date": [date(2026, 9, 28), date(2026, 9, 27)],
        "symbol": ["I'm great", "woo"],
        "open": [1.0, 1.0],
        "high": [2.5, 2.0],
        "low": [0.8, 0.5],
        "close": [2.0, 1.2],
        "volume": [100000, 200000],
    }
)


@pytest.fixture
def base_dict():
    return {
        "date": [date(2026, 9, 28)],
        "symbol": ["I'm great"],
        "open": [1.0],
        "high": [2.5],
        "low": [0.8],
        "close": [2.0],
        "volume": [100000],
    }


def test_validate_bars_orders():
    tab = validate_bars(df0)
    assert tab["date"].to_pylist() == [date(2026, 9, 27), date(2026, 9, 28)]


@pytest.mark.parametrize(
    "k,val,err",
    [
        ("date", "bad date", "date values could not be parsed"),
        ("symbol", 1, "Conversion failed"),
        ("open", "a", "Conversion failed"),
        ("high", "a", "Conversion failed"),
        ("low", "a", "Conversion failed"),
        ("close", "a", "Conversion failed"),
        ("volume", "1000.0", "Conversion failed"),
    ],
)
def test_invalid_types(base_dict, k: str, val, err: str):
    base_dict[k] = [val]
    df = DataFrame(base_dict)
    with pytest.raises(BarValidationError, match=err):
        validate_bars(df)


@pytest.mark.parametrize(
    "k",
    [("date"), ("symbol"), ("open"), ("high"), ("low"), ("close"), ("volume")],
)
def test_not_null_keys(base_dict, k: str):
    base_dict[k] = [None]
    df = DataFrame(base_dict)

    with pytest.raises(BarValidationError, match="was non-nullable"):
        validate_bars(df)


def test_no_repeating_keys():
    df = DataFrame(
        {
            "date": [date(2026, 9, 28), date(2026, 9, 28)],
            "symbol": ["a", "a"],
            "open": [1.0, 1.0],
            "high": [2.5, 2.0],
            "low": [0.8, 0.5],
            "close": [2.0, 1.2],
            "volume": [100000, 200000],
        }
    )

    with pytest.raises(BarValidationError, match="duplicate"):
        validate_bars(df)


@pytest.mark.parametrize(
    "k",
    [("open"), ("low")],
)
def test_prices_zero(base_dict, k: str):
    base_dict[k] = [0]
    df = DataFrame(base_dict)

    with pytest.raises(BarValidationError, match="> 0: 1 rows"):
        validate_bars(df)


@pytest.mark.parametrize("k", ["high", "close"])
def test_prices_zero_named_rule(base_dict, k: str):
    base_dict[k] = [0.0]
    if k == "high":
        base_dict["low"] = [0.0]
    df = DataFrame(base_dict)

    with pytest.raises(BarValidationError, match=f"{k} > 0: 1 rows"):
        validate_bars(df)


@pytest.mark.parametrize("k", ["open", "high", "low", "close"])
@pytest.mark.parametrize("val", [float("inf"), float("-inf")])
def test_prices_not_finite(base_dict, k: str, val: float):
    base_dict[k] = [val]
    df = DataFrame(base_dict)
    with pytest.raises(BarValidationError, match=f"{k} is finite: 1 rows"):
        validate_bars(df)


@pytest.mark.parametrize(
    "k",
    [("open"), ("low"), ("volume")],
)
def test_negative(base_dict, k: str):
    base_dict[k] = [-1]
    df = DataFrame(base_dict)
    with pytest.raises(
        BarValidationError, match=f"{k} >{'=' if k == 'volume' else ''} 0: 1 rows"
    ):
        validate_bars(df)


def test_prices_equal():
    dic = {
        "date": [date(2026, 9, 28)],
        "symbol": ["I'm great"],
        "open": [1.0],
        "high": [1.0],
        "low": [1.0],
        "close": [1.0],
        "volume": [100000],
    }
    df = DataFrame(dic)
    validate_bars(df)


def test_high_less_than_low():
    dic = {
        "date": [date(2026, 9, 28)],
        "symbol": ["I'm great"],
        "open": [1.0],
        "high": [0.5],
        "low": [1.0],
        "close": [1.0],
        "volume": [100000],
    }
    df = DataFrame(dic)
    with pytest.raises(BarValidationError, match="high >= low:"):
        validate_bars(df)


@pytest.mark.parametrize(
    "k",
    [("open"), ("close")],
)
def test_high_less_than_open_close(k: str):
    dic = {
        "date": [date(2026, 9, 28)],
        "symbol": ["I'm great"],
        "open": [1.5],
        "high": [1.5],
        "low": [1.0],
        "close": [1.5],
        "volume": [100000],
    }
    dic[k] = [2.0]
    df = DataFrame(dic)
    with pytest.raises(BarValidationError, match=re.escape("high >= max(open, close)")):
        validate_bars(df)


@pytest.mark.parametrize(
    "k",
    [("open"), ("close")],
)
def test_low_more_than_open_close(k: str):
    dic = {
        "date": [date(2026, 9, 28)],
        "symbol": ["I'm great"],
        "open": [1.5],
        "high": [1.5],
        "low": [1.0],
        "close": [1.5],
        "volume": [100000],
    }
    dic[k] = [0.5]
    df = DataFrame(dic)
    with pytest.raises(BarValidationError, match=re.escape("low <= min(open, close)")):
        validate_bars(df)


def test_datetime(base_dict):
    base_dict["date"] = [datetime(2026, 9, 28, 12, 1, 1)]
    df = DataFrame(base_dict)
    with pytest.raises(BarValidationError, match="1 date values have a time component"):
        validate_bars(df)


def test_date_with_timezone(base_dict):
    base_dict["date"] = [datetime(2026, 9, 28, tzinfo=timezone(timedelta(hours=5)))]
    df = DataFrame(base_dict)
    with pytest.raises(BarValidationError, match="date column is timezone-aware"):
        validate_bars(df)


@pytest.mark.parametrize(
    "ts,tz",
    [
        (Timestamp("2026-09-28 00:00", tz="UTC"), "UTC"),
        (Timestamp("2026-09-28 04:00", tz="UTC"), "America/New_York"),
    ],
)
def test_date_with_timezone_and_tz(base_dict, ts: Timestamp, tz: str):
    base_dict["date"] = [ts]
    df = DataFrame(base_dict)
    tab = validate_bars(df, tz=tz)
    assert tab["date"].to_pylist() == [date(2026, 9, 28)]


def test_iso_date_string_accepted(base_dict):
    base_dict["date"] = ["2026-09-28"]
    df = DataFrame(base_dict)
    tab = validate_bars(df)
    assert tab["date"].to_pylist() == [date(2026, 9, 28)]


@pytest.mark.parametrize("val", ["09/10/2026", "28/09/2026", "Sep 28 2026"])
def test_non_iso_date_string_rejected(base_dict, val: str):
    base_dict["date"] = [val]
    df = DataFrame(base_dict)
    with pytest.raises(DateParseError, match="could not be parsed"):
        validate_bars(df)


def test_duplicated_columns(base_dict):
    df = DataFrame(base_dict)
    df = concat([df, df[["volume"]]], axis=1)
    with pytest.raises(BarValidationError, match="duplicated columns:"):
        validate_bars(df)


def test_missing_columns(base_dict):
    del base_dict["volume"]
    df = DataFrame(base_dict)
    with pytest.raises(BarValidationError, match="missing columns:"):
        validate_bars(df)


def test_extra_columns(base_dict):
    base_dict["extra"] = ["hello there"]
    df = DataFrame(base_dict)
    with pytest.raises(BarValidationError, match="unexpected columns:"):
        validate_bars(df)
