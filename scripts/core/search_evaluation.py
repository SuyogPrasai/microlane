from typing import Any
import pandas as pd

def search_records(
    df: pd.DataFrame,
    fields: list[str] | None = None,
    **filters: Any,
) -> pd.DataFrame:
    result = df.copy()

    for column, value in filters.items():
        if value is None:
            continue

        if column not in result.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        result = result[result[column] == value]

    if fields:
        missing = [f for f in fields if f not in result.columns]
        if missing:
            raise ValueError(f"Fields not found: {missing}")
        result = result[fields]

    return result