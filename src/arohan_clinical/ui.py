"""Reusable Streamlit components and the dark theme."""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import CHART_SEQUENCE, COLORS

C = COLORS

CSS = f"""
<style>
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ color: {C['text']}; letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {C['border']}; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 8px; margin-bottom: 2px; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {C['elevated']}; border-left: 3px solid {C['primary']}; }}
.ar-header {{ border-bottom: 1px solid {C['border']}; padding-bottom: 14px; margin-bottom: 22px; }}
.ar-title {{ font-size: 1.85rem; font-weight: 750; color: {C['text']}; margin: 0; }}
.ar-sub {{ color: {C['muted']}; font-size: 0.98rem; margin-top: 4px; }}
.ar-card {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 12px; padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(0,0,0,.25); height: 100%; }}
.ar-label {{ color: {C['muted']}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.ar-value {{ color: {C['text']}; font-size: 1.7rem; font-weight: 750; margin-top: 4px; line-height: 1.15; }}
.ar-hint {{ color: {C['muted']}; font-size: .82rem; margin-top: 4px; }}
.ar-badge {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 650; border: 1px solid; }}
.ar-notice {{ border-radius: 10px; padding: 12px 16px; margin: 10px 0; border: 1px solid; background: {C['card']}; color: {C['text']}; }}
.ar-empty {{ background: {C['card']}; border: 1px dashed {C['border']}; border-radius: 12px; padding: 28px; color: {C['text']}; }}
.ar-table {{ width: 100%; border-collapse: collapse; background: {C['card']}; border: 1px solid {C['border']}; border-radius: 10px; overflow: hidden; }}
.ar-table th {{ text-align:left; color: {C['muted']}; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em;
               background: {C['elevated']}; padding: 10px 12px; border-bottom: 1px solid {C['border']}; }}
.ar-table td {{ color: {C['text']}; padding: 10px 12px; border-bottom: 1px solid {C['border']}; font-size: .9rem; vertical-align: top; }}
.ar-wrap {{ overflow-x: auto; }}
.ar-brand-name {{ font-weight: 800; letter-spacing: .12em; color: {C['text']}; font-size: 1.05rem; }}
.ar-brand-sub {{ color: {C['muted']}; font-size: .8rem; line-height: 1.4; }}
</style>
"""

TONES = {
    "success": C["success"], "warning": C["warning"], "error": C["error"], "primary": C["primary"],
    "secondary": C["secondary"], "muted": C["muted"],
}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def tone_for(status: str) -> str:
    s = str(status).lower()
    if s.startswith(("implemented", "yes", "passed", "available", "verified", "current")):
        return "success"
    if s.startswith(("pending", "warning", "stale", "partial", "not yet")):
        return "warning"
    if s.startswith(("unavailable", "no", "issues", "fail")):
        return "error"
    return "muted"


def badge_html(text: str, tone: str | None = None) -> str:
    colour = TONES.get(tone or tone_for(text), C["muted"])
    return (f'<span class="ar-badge" style="color:{colour};border-color:{colour};background:{colour}1A">'
            f"{html.escape(str(text))}</span>")


def page_header(title: str, subtitle: str, badge: str | None = "Research Prototype") -> None:
    b = badge_html(badge, "primary") if badge else ""
    st.markdown(f'<div class="ar-header"><div class="ar-title">{html.escape(title)} &nbsp;{b}</div>'
                f'<div class="ar-sub">{html.escape(subtitle)}</div></div>', unsafe_allow_html=True)


def metric_card_html(label: str, value, hint: str = "", tone: str | None = None) -> str:
    colour = TONES.get(tone, C["text"]) if tone else C["text"]
    return (f'<div class="ar-card"><div class="ar-label">{html.escape(label)}</div>'
            f'<div class="ar-value" style="color:{colour}">{html.escape(str(value))}</div>'
            f'<div class="ar-hint">{html.escape(hint)}</div></div>')


def metric_row(items: list[dict]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(metric_card_html(it["label"], it["value"], it.get("hint", ""), it.get("tone")), unsafe_allow_html=True)


def notice(text: str, tone: str = "warning", title: str | None = None) -> None:
    colour = TONES.get(tone, C["warning"])
    head = f"<strong>{html.escape(title)}</strong><br>" if title else ""
    st.markdown(f'<div class="ar-notice" style="border-color:{colour};border-left:4px solid {colour}">{head}{html.escape(text)}</div>',
                unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="ar-empty"><h3 style="margin-top:0">{html.escape(title)}</h3>'
                f'<div style="color:{C["muted"]}">{body}</div></div>', unsafe_allow_html=True)


def status_table(df: pd.DataFrame, status_cols: list[str]) -> None:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        cells = ""
        for c in df.columns:
            v = row[c]
            cells += f"<td>{badge_html(v)}</td>" if c in status_cols else f"<td>{html.escape(str(v))}</td>"
        body += f"<tr>{cells}</tr>"
    st.markdown(f'<div class="ar-wrap"><table class="ar-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',
                unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, title=title, paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        font=dict(color=C["text"], family="Segoe UI, Arial, sans-serif", size=13),
        margin=dict(l=40, r=20, t=50 if title else 20, b=40), colorway=CHART_SEQUENCE,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    return fig


def plot(fig: go.Figure, key: str, height: int = 360, title: str | None = None) -> None:
    st.plotly_chart(style_fig(fig, height, title), width="stretch", key=key)


def sidebar_brand(dataset_status: str) -> None:
    st.markdown(
        f'<div class="ar-card" style="margin-top:12px"><div class="ar-brand-name">AROHAN</div>'
        f'<div class="ar-brand-sub">Clinical Data Intelligence<br>Cardiovascular Disease Research</div>'
        f'<div style="margin-top:10px">{badge_html("Research Prototype", "primary")}</div>'
        f'<div class="ar-hint" style="margin-top:10px">Dataset: {badge_html(dataset_status)}</div></div>',
        unsafe_allow_html=True)
