from typing import Any
import pandas as pd


def search_records(
    df: pd.DataFrame,
    fields: list[str] | None = None,
    **filters: Any,
) -> list[pd.DataFrame]:
    """
    Search records from the dataframe with optional field selection and filters.

    Args:
        df       : the source DataFrame
        fields   : list of columns to return (None = all columns)
        **filters: column=value pairs to filter on

    Returns:
        list of single-row DataFrames
    """
    result = df.copy()

    for column, value in filters.items():
        if not value:
            continue

        if column not in result.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        result = result[result[column] == value]

    if fields:
        missing = [f for f in fields if f not in result.columns]
        if missing:
            raise ValueError(f"Fields not found: {missing}")
        result = result[fields]

    return [result.iloc[[i]] for i in range(len(result))]