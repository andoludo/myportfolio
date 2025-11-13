import itertools
from typing import Dict, TYPE_CHECKING

import pandas as pd  # type: ignore
from plotly import express as px, graph_objects as go

if TYPE_CHECKING:
    from myportfolio.models import PortfolioPoint

color_cycle = itertools.cycle(px.colors.qualitative.Plotly)


def format_hover(weights_dict: Dict[str, float]) -> str:
    items = [(k, v) for k, v in weights_dict.items() if v > 1e-6]
    items.sort(key=lambda kv: kv[1], reverse=True)
    lines = [
        f"{k}: {v:,.2%}" if k not in ["beta", "sharpe_ratio"] else f"{k}:{round(v,2)}"
        for k, v in items
    ]
    return "<br>".join(lines)


def add_point(fig: go.Figure, point: "PortfolioPoint", name: str) -> go.Figure:
    fig.add_trace(
        go.Scatter(
            x=[point.volatility],
            y=[point.expecter_return],
            mode="markers",
            name=name,
            marker={"size": 10, "symbol": "x"},
            line={"color": next(color_cycle)},
            hovertext=[
                format_hover(
                    point.weights
                    | {
                        "beta": (point.beta or 0),
                        "sharpe_ratio": (point.sharpe_ratio or 0),
                    }
                )
            ],
            hovertemplate="Volatility: %{x:.2%}<br>Expected Return: %{y:.2%}<br><br>%{hovertext}<extra></extra>",
        )
    )
    return fig


def plot(fig: go.Figure, data: pd.DataFrame, name: str) -> go.Figure:
    hover_texts = [format_hover(w) for w in data["weight"]]
    fig.add_trace(
        go.Scatter(
            x=data.volatility,
            y=data["return"],
            mode="lines+markers",
            name=f"Efficient Frontier {name}",
            hovertext=hover_texts,
            line={"color": next(color_cycle)},
            hovertemplate="Volatility: %{x:.2%}<br>Expected Return: %{y:.2%}<br><br>%{hovertext}<extra></extra>",
        )
    )
    return fig
