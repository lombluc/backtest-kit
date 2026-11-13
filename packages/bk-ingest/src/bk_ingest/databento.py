from datetime import date
from typing import Any

import databento as db

from bk_ingest.exceptions import TooExpensiveException

client = db.Historical()

MAX_COST = 0.1


def get_coverage(dataset: str) -> dict[str, str | Any]:
    return client.metadata.get_dataset_range(dataset)


def estimate_cost(symbols: list[str], start: date, end: date) -> float:
    cost = client.metadata.get_cost(
        "XNAS.ITCH", start, end, symbols=symbols, schema="OHLCV_1D"
    )
    if cost > MAX_COST:
        raise TooExpensiveException(MAX_COST, cost)
    return cost


def fetch_raw(symbols: list[str], start: date, end: date):
    client.timeseries.get_range(
        "XNAS.ITCH", start, end, symbols=symbols, schema="OHLCV_1D"
    )
