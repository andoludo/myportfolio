from pathlib import Path
from typing import List

from bearish.database.crud import BearishDb

from bearish.models.price.prices import Prices  # type: ignore
from bearish.models.query.query import AssetQuery, Symbols  # type: ignore
from bearish.types import TickerOnlySources  # type: ignore

import plotly.graph_objects as go

from myportfolio.models import (
    Market,
    NewAsset,
    Asset,
    TimePeriodValue,
    TimePeriodData,
    Portfolio,
    PortfolioDescription,
    portfolio_optimize,
)


def test_portfolio(bear_db: Path) -> None:
    assets = [
        Asset(symbol="NVDA", value=2070),
        Asset(symbol="GOOG", value=1098),
        Asset(symbol="MSFT", value=1624),
        Asset(symbol="UNH", value=1421),
        Asset(symbol="LLY", value=2440),
        Asset(symbol="LMT", value=1941),
        Asset(symbol="MRK", value=1731),
    ]
    market = Market(symbol="^GSPC")
    bearish_db = BearishDb(database_path=bear_db)
    portfolio = Portfolio(
        assets=assets,
        market=market,
        bearish_db=bearish_db,
        time_period=TimePeriodValue(type="years", value=5),
    )
    kpi = portfolio.compute_kpi()
    assert kpi


def test_new_portfolio(bear_db: Path) -> None:
    assets = [
        NewAsset(symbol="NVDA"),
        NewAsset(symbol="GOOG"),
        NewAsset(symbol="MSFT"),
        NewAsset(symbol="UNH"),
        NewAsset(symbol="LLY"),
        NewAsset(symbol="LMT"),
        NewAsset(symbol="MRK"),
    ]
    market = Market(symbol="^GSPC")
    bearish_db = BearishDb(database_path=Path(bear_db))
    portfolio = Portfolio(
        assets=assets,
        market=market,
        bearish_db=bearish_db,
        value=5000,
        time_period=TimePeriodValue(type="years", value=1),
    )
    optimized_portfolio = portfolio.max_sharpe()

    assert optimized_portfolio


def test_portfolio_frontier(bear_db: Path) -> None:
    assets = [
        Asset(symbol="NVDA", value=2070),
        Asset(symbol="GOOG", value=1098),
        Asset(symbol="MSFT", value=1624),
        Asset(symbol="UNH", value=1421),
        Asset(symbol="LLY", value=2440),
        Asset(symbol="LMT", value=1941),
        Asset(symbol="MRK", value=1731),
    ]
    market = Market(symbol="^GSPC")
    bearish_db = BearishDb(database_path=Path(bear_db))
    figure = go.Figure()
    for time_period in [
        TimePeriodValue(type="years", value=5),
        TimePeriodValue(type="years", value=1),
    ]:
        portfolio = Portfolio(
            assets=assets, market=market, bearish_db=bearish_db, time_period=time_period
        )
        kpi = portfolio.compute_kpi()

        figure = kpi.plot(figure, "new")
    figure.show()
    assert True


def test_add_optimized(bear_db: Path) -> None:
    assets = [
        Asset(symbol="NVDA", value=2070),
        Asset(symbol="GOOG", value=1098),
        Asset(symbol="MSFT", value=1624),
        Asset(symbol="UNH", value=1421),
        Asset(symbol="LLY", value=2440),
        Asset(symbol="WMT", value=1024),
        Asset(symbol="LMT", value=1941),
        Asset(symbol="MRK", value=1731),
    ]
    bearish_db = BearishDb(database_path=bear_db)
    new_assets = [
        NewAsset(symbol="VZ"),
        NewAsset(symbol="TSLA"),
        NewAsset(symbol="AAPL"),
    ]
    portfolio_description = PortfolioDescription(
        current_assets=assets, new_assets=new_assets, amount=5000
    )
    figure = portfolio_optimize(bearish_db, portfolio_description)
    figure.show()
    assert figure


def test_only_new_assets(bear_db: Path) -> None:
    bearish_db = BearishDb(database_path=bear_db)
    new_assets = [
        NewAsset(symbol="VZ"),
        NewAsset(symbol="TSLA"),
        NewAsset(symbol="AAPL"),
    ]
    portfolio_description = PortfolioDescription(new_assets=new_assets, amount=5000)
    figure = portfolio_optimize(bearish_db, portfolio_description)
    figure.show()
    assert figure
