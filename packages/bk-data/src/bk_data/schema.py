from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from bk_data.exceptions import BarValidationError, DateParseError

schema_fields: list[pa.Field[Any]] = [
    pa.field("date", pa.date32(), nullable=False),
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("open", pa.float64(), nullable=False),
    pa.field("high", pa.float64(), nullable=False),
    pa.field("low", pa.float64(), nullable=False),
    pa.field("close", pa.float64(), nullable=False),
    pa.field("volume", pa.int64(), nullable=False),
]

bar_schema = pa.schema(schema_fields)

o, h, l_, c, v = (pc.field(n) for n in ["open", "high", "low", "close", "volume"])

BAR_RULES: dict[str, pc.Expression] = {
    "open is finite": pc.is_finite(o),
    "high is finite": pc.is_finite(h),
    "low is finite": pc.is_finite(l_),
    "close is finite": pc.is_finite(c),
    "open > 0": o > 0,
    "high > 0": h > 0,
    "low > 0": l_ > 0,
    "close > 0": c > 0,
    "volume >= 0": v >= 0,
    "high >= low": h >= l_,
    "high >= max(open, close)": h >= pc.max_element_wise(o, c, skip_nulls=True),
    "low <= min(open, close)": l_ <= pc.min_element_wise(o, c, skip_nulls=True),
}


def validate_bars(df: pd.DataFrame, tz: str | None = None) -> pa.Table:
    check_columns(df, bar_schema)
    df = normalize_dates(df, tz=tz)
    try:
        table = pa.Table.from_pandas(df, schema=bar_schema, preserve_index=False)

    except (ValueError, pa.ArrowInvalid, pa.ArrowTypeError) as err:
        raise BarValidationError(f"could not convert bars to Arrow: {err}") from err

    keys = ["date", "symbol"]
    # for k in keys:
    #     n_null = table[k].null_count
    #     if n_null:
    #         raise BarValidationError(f"column {k!r} has {n_null} null/NaN values")

    counts = table.group_by(keys).aggregate([([], "count_all")])
    if counts.num_rows != table.num_rows:
        dups = counts.filter(pc.greater(counts["count_all"], pa.scalar(1)))
        raise BarValidationError(
            f"{dups.num_rows} duplicate (date, symbol) keys, e.g.:\n"
            f"{dups.slice(0, 5).to_pandas()}"
        )

    check_rules(table, BAR_RULES)

    return table.sort_by([("date", "ascending"), ("symbol", "ascending")])


def check_columns(df: pd.DataFrame, schema: pa.Schema) -> None:
    expected = schema.names
    actual = list(df.columns)

    duplicated = sorted(set(df.columns[df.columns.duplicated()]))
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]

    problems = []
    if duplicated:
        problems.append(f"duplicated columns: {duplicated}")
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"unexpected columns: {extra}")
    if problems:
        raise BarValidationError("; ".join(problems))


def normalize_dates(df: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    # ISO8601 only, so ambiguous strings like "09/10/2026" are rejected rather
    # than silently read month-first. date/datetime objects are unaffected.
    ts = pd.to_datetime(df["date"], errors="coerce", format="ISO8601")
    failed = df["date"].notna() & ts.isna()
    if failed.any():
        raise DateParseError(df.loc[failed, ["date", "symbol"]])

    if ts.dt.tz is not None:
        if tz is None:
            raise BarValidationError(
                "date column is timezone-aware; pass tz= to say which calendar "
                "day to use"
            )
        ts = ts.dt.tz_convert(tz).dt.tz_localize(None)

    has_time = ts.notna() & (ts != ts.dt.normalize())
    if has_time.any():
        examples = df.loc[has_time, ["date", "symbol"]].head(5)
        raise BarValidationError(
            f"{has_time.sum()} date values have a time component, e.g.\n{examples}"
        )

    return df.assign(date=ts)


def check_rules(
    table: pa.Table,
    rules: dict[str, pc.Expression],
    null_is_violation: bool = False,
    show: int = 5,
) -> None:
    errors = []
    for name, expr in rules.items():
        bad = ~expr
        if null_is_violation:
            bad = bad | expr.is_null()
        violations = table.filter(bad)
        if violations.num_rows:
            sample = violations.select(["date", "symbol"]).slice(0, show).to_pandas()
            errors.append(f"{name}: {violations.num_rows} rows, e.g.\n{sample}")
    if errors:
        raise BarValidationError("\n\n".join(errors))
