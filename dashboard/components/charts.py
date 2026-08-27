"""Plotly chart helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEAL = "#008b8b"
AMBER = "#d99000"
GRAY = "#65758b"
GREEN = "#188a42"
RED = "#c0392b"


PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def apply_responsive_layout(fig, title: str | None = None):
    layout = {
        "autosize": True,
        "template": "plotly_white",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": dict(color="#1f2933", family="Arial, sans-serif"),
        "margin": dict(l=20, r=20, t=64, b=84),
        "legend": dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        "xaxis": dict(showgrid=True, gridcolor="#edf2f7", zeroline=False),
        "yaxis": dict(showgrid=True, gridcolor="#edf2f7", zeroline=False),
    }
    if title is not None:
        layout["title"] = dict(text=title, font=dict(color="#1f2933"))
    fig.update_layout(**layout)
    return fig


def empty_figure(message: str = "Sin datos"):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(height=280)
    return apply_responsive_layout(fig)


def forecast_line(df: pd.DataFrame, title: str = "Forecast diario"):
    if df.empty:
        return empty_figure()
    plot = df.groupby("forecast_date", observed=True)["forecast_units"].sum().reset_index()
    fig = px.line(plot, x="forecast_date", y="forecast_units", title=title, markers=True)
    fig.update_traces(line_color=TEAL)
    fig.update_layout(yaxis_title="Unidades", xaxis_title="Fecha")
    return apply_responsive_layout(fig)


def forecast_with_uncertainty(history: pd.DataFrame, forecast: pd.DataFrame):
    fig = go.Figure()
    if not history.empty:
        hist = history.groupby("date", observed=True)["demand"].sum().reset_index()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["demand"], mode="lines", name="Histórico", line=dict(color=GRAY)))
    if not forecast.empty:
        f = forecast.sort_values("forecast_date")
        fig.add_trace(go.Scatter(x=f["forecast_date"], y=f["forecast_upper_95"], mode="lines", name="Límite 95% sup.", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=f["forecast_date"], y=f["forecast_lower_95"], mode="lines", name="Banda empírica 95%", fill="tonexty", line=dict(width=0), fillcolor="rgba(0,139,139,.16)"))
        fig.add_trace(go.Scatter(x=f["forecast_date"], y=f["forecast_units"], mode="lines+markers", name="Forecast oficial", line=dict(color=TEAL)))
        fig.add_vline(x=f["forecast_date"].min(), line_dash="dash", line_color=AMBER)
    fig.update_layout(yaxis_title="Unidades", xaxis_title="Fecha")
    return apply_responsive_layout(fig, "Histórico reciente y forecast oficial")


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, orientation: str = "v"):
    if df.empty or x not in df or y not in df:
        return empty_figure()
    fig = px.bar(df, x=x, y=y, title=title, orientation=orientation, color_discrete_sequence=[TEAL])
    return apply_responsive_layout(fig)


def donut(df: pd.DataFrame, names: str, values: str, title: str):
    if df.empty:
        return empty_figure()
    fig = px.pie(df, names=names, values=values, hole=.55, title=title, color_discrete_sequence=[TEAL, AMBER, GREEN, RED, GRAY])
    return apply_responsive_layout(fig)


def model_metric_bars(df: pd.DataFrame, metric: str, title: str):
    if df.empty or metric not in df:
        return empty_figure()
    plot = df.sort_values(metric)
    fig = px.bar(plot, x="model", y=metric, title=title, color="model", color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, xaxis_tickangle=-25)
    return apply_responsive_layout(fig)


def inventory_projection(df: pd.DataFrame, title: str = "Proyección diaria de inventario"):
    if df.empty:
        return empty_figure()
    plot = df.groupby("date", observed=True).agg(projected_inventory_after_order=("projected_inventory_after_order", "sum"), stockout_units=("stockout_units", "sum"), recommended_order=("recommended_order", "sum")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot["date"], y=plot["projected_inventory_after_order"], mode="lines", name="Inventario proyectado", line=dict(color=TEAL)))
    fig.add_trace(go.Bar(x=plot["date"], y=plot["recommended_order"], name="Orden recomendada", marker_color=AMBER, opacity=.55))
    fig.add_trace(go.Bar(x=plot["date"], y=plot["stockout_units"], name="Faltante", marker_color=RED, opacity=.45))
    fig.update_layout(barmode="overlay", yaxis_title="Unidades")
    return apply_responsive_layout(fig, title)
