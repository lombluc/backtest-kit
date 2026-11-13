class TooExpensiveException(Exception):
    def __init__(self, max_cost: float, cost: float):
        super().__init__(
            f"Estimated cost of API call is ${cost}"
            f"which is more than the given max cost ${max_cost}."
        )
