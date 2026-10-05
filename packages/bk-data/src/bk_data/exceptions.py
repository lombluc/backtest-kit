from pandas import DataFrame


class BarValidationError(ValueError):
    """Raised when a bars DataFrame doesn't meet the expected contract."""


class DateParseError(BarValidationError):
    def __init__(self, bad_rows: DataFrame):
        self.bad_rows = bad_rows
        super().__init__(
            f"{len(bad_rows)} date values could not be parsed, e.g.\n"
            f"{bad_rows.head(5).to_string()}"
        )
