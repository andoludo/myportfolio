from datetime import date
from functools import cached_property
from typing import Optional, Dict, List, Self

import numpy as np
import pandas as pd  # type: ignore
from bearish.database.crud import BearishDb  # type: ignore
from bearish.database.schemas import PriceIndexORM  # type: ignore
from plotly import graph_objects as go
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pypfopt import expected_returns, risk_models, EfficientFrontier  # type: ignore

from myportfolio.types import TimePeriod
from myportfolio.utils import _read_series
from myportfolio.services import add_point, plot


class BaseBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseAsset(BaseBaseModel):
    symbol: str
    data: pd.Series = Field(default_factory=pd.Series)

    def valid(self) -> bool:
        return not self.data.empty

    def init(
        self,
        bearish_db: BearishDb,
        time_period: "TimePeriodValue",
        total_value: Optional[float] = None,
        months: int = 12 * 5,
    ) -> None:
        data = self._read_data(bearish_db=bearish_db, months=months)
        latest_date = data.index.max()
        if not data.empty:
            self.data = data[
                data.index >= latest_date - pd.DateOffset(**time_period._to_dict())
            ].copy()

        if total_value:
            self._compute_weight(total_value)

    def __hash__(self) -> int:
        return hash(self.symbol)

    def _read_data(self, bearish_db: BearishDb, months: int = 12 * 5) -> pd.Series:
        return _read_series(bearish_db, self.symbol, months=months)

    def _compute_weight(self, total_value: float) -> None: ...
    def compute_beta(self, market_data: pd.Series) -> float:
        raise NotImplementedError()


class Market(BaseAsset):
    def _read_data(self, bearish_db: BearishDb, months: int = 12 * 5) -> pd.Series:
        return _read_series(bearish_db, self.symbol, months=months, table=PriceIndexORM)


class NewAsset(BaseAsset):
    value: Optional[float] = None


class Asset(BaseAsset):
    value: float
    weight: Optional[float] = None
    position_open_date: Optional[date] = None
    position_close_date: Optional[date] = None
    beta: Optional[float] = None

    def _compute_weight(self, total_value: float) -> None:
        if self.weight is None:
            self.weight = self.value / total_value

    def compute_beta(self, market_data: pd.Series) -> float:
        asset_returns = np.log(self.data / self.data.shift(1)).resample("M").mean()
        market_returns = np.log(market_data / market_data.shift(1)).resample("M").mean()
        asset_returns = asset_returns.loc[market_returns.index]
        cov_with_market = asset_returns.cov(market_returns)
        market_var = market_returns.var()
        beta = cov_with_market / market_var
        if self.beta is None:
            self.beta = beta
        return float(beta)


class TimePeriodValue(BaseBaseModel):
    type: TimePeriod
    value: int

    def __hash__(self) -> int:
        return hash((self.type, self.value))

    def __eq__(self, other: object) -> bool:
        return bool(self.type == other.type and self.value == other.value)  # type: ignore

    def _to_dict(self) -> Dict[str, int]:
        return {self.type: self.value}


class TimePeriodData(TimePeriodValue):
    data: pd.DataFrame = Field(default_factory=pd.DataFrame)

    def update(self, df: pd.DataFrame) -> None:
        latest_date = df.index.max()
        self.data = df[
            df.index >= latest_date - pd.DateOffset(**self._to_dict())
        ].copy()

    def time_period(self) -> TimePeriodValue:
        return TimePeriodValue(type=self.type, value=self.value)


class PortfolioData(BaseBaseModel):
    assets: pd.DataFrame = Field(default_factory=pd.DataFrame)
    market: pd.DataFrame = Field(default_factory=pd.DataFrame)


class PortfolioPoint(BaseBaseModel):
    expecter_return: float
    volatility: float
    weights: Dict[str, float] = Field(default_factory=dict)
    sharpe_ratio: Optional[float] = None
    beta: Optional[float] = None


class PortfolioKpi(PortfolioPoint):
    time_period: TimePeriodValue
    sharpe_ratio: float
    efficient_frontier: pd.DataFrame = Field(default_factory=pd.DataFrame)

    def name(self) -> str:
        return f"{self.time_period.value}-{self.time_period.type}"

    def to_point(self) -> PortfolioPoint:
        return PortfolioPoint(
            expecter_return=self.expecter_return,
            volatility=self.volatility,
            weights=self.weights,
            beta=self.beta,
            sharpe_ratio=self.sharpe_ratio,
        )

    def plot(self, fig: go.Figure, name: str) -> go.Figure:
        if not self.efficient_frontier.empty:
            fig = plot(fig, self.efficient_frontier, name=f"{name} ({self.name()})")
            fig = add_point(fig, self.to_point(), name=f"{name} ({self.name()})")
        return fig


class OptimizedPortfolioKpi(PortfolioKpi):
    assets: List[Asset]


class Portfolio(BaseModel):
    time_period: TimePeriodValue
    bearish_db: BearishDb
    assets: List[Asset] | List[NewAsset]
    market: Market
    data: PortfolioData = Field(default_factory=PortfolioData)
    value: Optional[float] = None

    def add(self, optimized_portfolio: OptimizedPortfolioKpi) -> "Portfolio":
        if not self.time_period == optimized_portfolio.time_period:
            raise ValueError(
                "time_period must be equal to optimized_portfolio.time_period"
            )
        new_assets = self.assets + optimized_portfolio.assets
        return Portfolio(
            time_period=self.time_period,
            assets=new_assets,
            market=self.market,
            bearish_db=self.bearish_db,
        )

    def _merge_assets(self) -> None:
        if all(isinstance(a, Asset) for a in self.assets):
            self.assets = [
                Asset(
                    symbol=symbol,
                    value=sum(
                        [ass.value for ass in self.assets if ass.symbol == symbol]
                    ),
                )
                for symbol in {a.symbol for a in self.assets}
            ]

    @model_validator(mode="after")  # type: ignore
    def _validate(self) -> None:
        self._merge_assets()
        if all(isinstance(a, NewAsset) for a in self.assets) and self.value is None:
            raise ValueError()
        if all(isinstance(a, Asset) for a in self.assets):
            self.value = sum([a.value for a in self.assets])  # type: ignore
        for asset in self.assets:
            asset.init(
                bearish_db=self.bearish_db,
                time_period=self.time_period,
                total_value=self.value,
            )

        self.market.init(
            bearish_db=self.bearish_db,
            time_period=self.time_period,
            total_value=self.value,
        )
        asset_data = pd.concat(
            [s.data for s in self.assets if not s.data.empty], axis=1
        )
        self.data.assets = asset_data
        self.data.market = self.market.data

    def get_weights(self) -> Dict[str, float]:
        return {
            a.symbol: a.weight for a in self.assets if isinstance(a, Asset) and a.weight
        }

    def _compute_beta(self, market_data: pd.Series) -> float:
        beta = 0
        for asset in self.assets:
            if not asset.valid():
                continue
            if not isinstance(asset, Asset):
                continue
            beta += asset.weight * asset.compute_beta(market_data)  # type: ignore
        return beta

    @cached_property
    def _capm_expected_return(self) -> pd.Series:
        return expected_returns.capm_return(
            self.data.assets, market_prices=self.market.data, risk_free_rate=0.04
        )

    @cached_property
    def _risks(self) -> np.ndarray:  # type: ignore
        return risk_models.CovarianceShrinkage(self.data.assets).ledoit_wolf()  # type: ignore

    def _efficient_frontier(self) -> pd.DataFrame:

        target_returns = np.linspace(
            self._capm_expected_return.min(), self._capm_expected_return.max(), 100
        )
        vols, rets, weight_list = [], [], []
        weight_bounds = (0, 1)
        for r in target_returns:
            ef = EfficientFrontier(
                self._capm_expected_return, self._risks, weight_bounds=weight_bounds
            )
            try:
                ef.efficient_return(target_return=float(r))
                ret, vol, _ = ef.portfolio_performance(verbose=False)
                w = ef.clean_weights()
                vols.append(vol)
                rets.append(ret)
                weight_list.append(w)
            except Exception:  # noqa: S112
                continue
        return pd.DataFrame({"return": rets, "volatility": vols, "weight": weight_list})

    def compute_kpi(self) -> PortfolioKpi:
        weights = self.get_weights()
        if not weights:
            raise ValueError("No weights provided")
        ef = EfficientFrontier(self._capm_expected_return, self._risks)
        ef.set_weights(weights)
        ret, vol, sharpe = ef.portfolio_performance(verbose=True)
        beta = self._compute_beta(self.market.data)
        efficient_frontier = self._efficient_frontier()
        return PortfolioKpi(
            weights=weights,
            expecter_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            beta=beta,
            time_period=self.time_period,
            efficient_frontier=efficient_frontier,
        )

    def max_sharpe(self) -> OptimizedPortfolioKpi:
        weight_bounds = (0, 1)
        ef = EfficientFrontier(
            self._capm_expected_return, self._risks, weight_bounds=weight_bounds
        )
        ef.max_sharpe()
        ret, vol, sharpe = ef.portfolio_performance(verbose=False)
        w = ef.clean_weights()
        assets = [
            Asset(symbol=symbol, value=weight * self.value)
            for symbol, weight in w.items()
        ]
        return OptimizedPortfolioKpi(
            weights=w,
            expecter_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            assets=assets,
            time_period=self.time_period,
        )


class PortfolioDescription(BaseModel):
    current_assets: Optional[List[Asset]] = None
    new_assets: Optional[List[NewAsset]] = None
    market: Market = Field(default_factory=lambda: Market(symbol="^GSPC"))
    time_period: TimePeriodValue = Field(
        default_factory=lambda: TimePeriodValue(type="years", value=1)
    )
    amount: Optional[float] = None

    @model_validator(mode="after")
    def _portfolio_validator(self) -> Self:
        if self.new_assets and self.amount is None:
            raise ValueError("Amount must be provided")
        return self


def portfolio_optimize(
    bearish_db: BearishDb,
    portfolio_description: "PortfolioDescription",
) -> go.Figure:
    figure = go.Figure()
    portfolio: Optional[Portfolio] = None
    for time_period in [
        TimePeriodValue(type="years", value=1),
        TimePeriodValue(type="years", value=5),
        TimePeriodValue(type="months", value=6),
    ]:
        if portfolio_description.current_assets:
            portfolio = Portfolio(
                assets=portfolio_description.current_assets,
                market=portfolio_description.market,
                bearish_db=bearish_db,
                time_period=time_period,
            )
            kpi = portfolio.compute_kpi()
            figure = kpi.plot(figure, "Current")
        if portfolio_description.new_assets:
            portfolio_new = Portfolio(
                assets=portfolio_description.new_assets,
                market=portfolio_description.market,
                bearish_db=bearish_db,
                value=portfolio_description.amount,
                time_period=time_period,
            )
            optimized_portfolio = portfolio_new.max_sharpe()
            if portfolio:
                portfolio_final = portfolio.add(optimized_portfolio)
                kpi = portfolio_final.compute_kpi()
            else:
                new_portfolio = Portfolio(
                    assets=optimized_portfolio.assets,
                    market=portfolio_description.market,
                    bearish_db=bearish_db,
                    time_period=time_period,
                )
                kpi = new_portfolio.compute_kpi()
            figure = kpi.plot(figure, "New")
    figure.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility (Standard Deviation)",
        yaxis_title="Expected Return",
        template="plotly_white",
        font={"size": 14},
        hovermode="closest",
        margin={"l": 70, "r": 30, "t": 60, "b": 60},
        width=1200,
        height=900,
    )

    figure.update_xaxes(tickformat=".1%")
    figure.update_yaxes(tickformat=".1%")
    return figure
