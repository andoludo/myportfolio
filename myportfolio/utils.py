from typing import Optional, Type

import pandas as pd  # type: ignore
from bearish.database.crud import BearishDb  # type: ignore
from bearish.models.base import Ticker  # type: ignore
from bearish.models.price.prices import Prices  # type: ignore
from bearish.models.query.query import AssetQuery, Symbols  # type: ignore
from sqlmodel import SQLModel


def _read_series(
    bearish_db: BearishDb,
    symbol: str,
    months: int = 12 * 5,
    table: Optional[Type[SQLModel]] = None,
) -> pd.Series:
    prices = bearish_db.read_series(
        AssetQuery(symbols=Symbols(index=[Ticker(symbol=symbol)])),
        months=months,
        table=table,
    )
    data = Prices(prices=prices).to_dataframe()
    if data.empty or "close" not in data.columns:
        return pd.Series()
    return data["close"].rename(symbol)
