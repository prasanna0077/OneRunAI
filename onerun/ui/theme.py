"""OneRun AI look & feel (dark slate + cyan, taken from the pitch deck) and small UI helpers."""
from __future__ import annotations

import io
from urllib.parse import quote

import pandas as pd
import streamlit as st

from onerun.config import C, PRODUCT_NAME, PRODUCT_SUBTITLE, SLOGAN, STAGE_COLOURS

# decorative "radar dial" motif used on every slide of the deck
_DIAL = ("<svg xmlns='http://www.w3.org/2000/svg' width='460' height='460' viewBox='0 0 460 460' fill='none' "
         "stroke='#7d7d7d' stroke-width='1.4'><circle cx='230' cy='230' r='215'/><circle cx='230' cy='230' r='185' "
         "stroke-dasharray='2 5'/><circle cx='230' cy='230' r='150'/><path d='M230 80 A150 150 0 0 1 380 230' "
         "stroke-width='4' stroke='#9a9a9a'/><circle cx='230' cy='230' r='95' stroke-dasharray='6 6'/></svg>")

_CSS = """
.stApp { background: __BG__; }
.stApp::before { content:""; position:fixed; top:-150px; right:-150px; width:460px; height:460px; opacity:.35;
  background:url("data:image/svg+xml,__DIAL__") no-repeat; pointer-events:none; z-index:0; }
[data-testid="stSidebar"] { background: __DEEP__; border-right:1px solid #333; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; max-width: 1320px; }
h2, h3, h4 { color: __PRIMARY__ !important; }
hr { border-color: __BORDER__; }

.or-logo { display:inline-flex; align-items:center; justify-content:center; border:3px solid __LOGO__; border-radius:14px;
  padding:0 14px; font-weight:800; color:#fff; background:__DEEP__; letter-spacing:-1px; }
.or-brand { display:flex; align-items:center; gap:14px; margin-bottom:4px; }
.or-brand .name { font-size:26px; font-weight:700; color:#fff; line-height:1.05; }
.or-brand .sub { font-size:14px; color:__PRIMARY__; line-height:1.2; }
.or-brand .slogan { font-size:12px; color:#d0d0d0; }

.or-card { background:__CARD__; border:1px solid __BORDER__; border-radius:12px; padding:16px 18px; height:100%; }
.or-card h4 { margin:0 0 6px 0; font-size:16px; }
.or-card p, .or-card li { color:#d9d9d9; font-size:14px; line-height:1.5; }
.or-kpi { background:__CARD__; border:1px solid __BORDER__; border-left:4px solid var(--accent,__PRIMARY__);
  border-radius:10px; padding:12px 14px; }
.or-kpi .l { font-size:11.5px; text-transform:uppercase; letter-spacing:.6px; color:__MUTED__; }
.or-kpi .v { font-size:26px; font-weight:700; color:#fff; line-height:1.2; }
.or-kpi .s { font-size:12px; color:__MUTED__; }
.or-chip { display:inline-block; padding:2px 11px; border-radius:999px; font-size:11.5px; font-weight:700;
  letter-spacing:.4px; color:#111; margin-right:6px; }
.or-problem { font-size:16px; color:#fff; margin:2px 0 10px 0; }
.or-problem b { color:__RED__; }
.or-label { color:__PRIMARY__; font-weight:700; font-size:15px; margin:10px 0 4px 0; }
.or-check { color:#e6e6e6; font-size:14px; margin:2px 0; }
.or-check:before { content:"\\2713  "; color:__GREEN__; font-weight:700; }
.or-flow { display:flex; align-items:stretch; gap:10px; }
.or-flow .step { flex:1; border-radius:12px; padding:14px 16px; color:#111; }
.or-flow .arrow { align-self:center; font-size:30px; color:#888; }
.or-flow .step h4 { color:#111 !important; margin:0 0 4px 0; } .or-flow .step li { color:#1a1a1a; font-size:13.5px; }
table.or-table { width:100%; border-collapse:collapse; font-size:13.5px; }
table.or-table th { background:__THEAD__; color:#111; padding:6px 10px; text-align:left; border:1px solid #777; }
table.or-table td { padding:5px 10px; border:1px solid #666; color:#eee; }
.or-layer { border:1px dashed __PRIMARY__; border-radius:12px; padding:10px 14px; margin-bottom:10px; background:#2b2b2b; }
.or-layer .t { color:__PRIMARY__; font-weight:700; font-size:13px; text-transform:uppercase; letter-spacing:.6px; }
.or-box { display:inline-block; background:#1f1f1f; border:1px solid __BORDER__; border-radius:8px; padding:5px 11px;
  margin:4px 6px 2px 0; font-size:13px; color:#eee; }
.or-scale { position:relative; height:9px; border-radius:6px; overflow:hidden; display:flex; margin:8px 0 2px 0; }
.or-scale span { display:block; height:100%; }
.or-scale-wrap { position:relative; }
.or-marker { position:absolute; top:-3px; width:3px; height:15px; background:#fff; border-radius:2px; box-shadow:0 0 0 1px #000; }
.or-refline { position:absolute; top:-4px; width:0; height:17px; border-left:2px dashed #fff; opacity:.75; }
.or-scale-lbl { display:flex; justify-content:space-between; font-size:10.5px; color:__MUTED__; }
.or-rule { font-size:14px; margin:3px 0; color:#e6e6e6; }
.or-rule .ok { color:__GREEN__; font-weight:700; } .or-rule .no { color:__RED__; font-weight:700; }
.or-legend td.sw { width:14px; }
@media (max-width: 900px) { .or-flow { flex-direction:column; } .or-flow .arrow { transform:rotate(90deg); } }
"""


def apply_theme() -> None:
    css = (_CSS.replace("__DIAL__", quote(_DIAL)).replace("__BG__", C.BG).replace("__DEEP__", C.BG_DEEP)
           .replace("__PRIMARY__", C.PRIMARY).replace("__BORDER__", C.BORDER).replace("__CARD__", C.CARD)
           .replace("__LOGO__", C.LOGO_BORDER).replace("__MUTED__", C.MUTED).replace("__RED__", C.RED)
           .replace("__GREEN__", C.GREEN).replace("__THEAD__", C.TABLE_HEAD))
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- html snippets
def logo_html(size: int = 34) -> str:
    return f"<span class='or-logo' style='font-size:{size}px;line-height:{int(size * 1.3)}px'>1R</span>"


def brand_block(compact: bool = False) -> str:
    size = 24 if compact else 40
    slogan = "" if compact else f"<div class='slogan'>{SLOGAN}</div>"
    return (f"<div class='or-brand'>{logo_html(size)}<div><div class='name' style='font-size:{18 if compact else 28}px'>{PRODUCT_NAME}</div>"
            f"<div class='sub'>{PRODUCT_SUBTITLE}</div>{slogan}</div></div>")


def chip(text: str, colour: str = C.PRIMARY) -> str:
    return f"<span class='or-chip' style='background:{colour}'>{text}</span>"


def scale_bar(value: float, bands: list, vmin: float = 0, vmax: float = 100, ref: float | None = None) -> str:
    """Segmented bar with a marker at `value`. bands = [(upper_edge, colour, label), ...] in ascending order."""
    span = float(vmax - vmin)
    pct = lambda v: max(0.0, min(100.0, (v - vmin) / span * 100))      # noqa: E731
    segs, lo, labels = "", vmin, ""
    for upper, colour, name in bands:
        w = pct(upper) - pct(lo)
        segs += f"<span style='width:{w:.2f}%;background:{colour}'></span>"
        labels += f"<span>{name}</span>"
        lo = upper
    refline = f"<div class='or-refline' style='left:{pct(ref):.2f}%' title='Reference'></div>" if ref is not None else ""
    return (f"<div class='or-scale-wrap'><div class='or-scale'>{segs}</div>{refline}"
            f"<div class='or-marker' style='left:calc({pct(value):.2f}% - 1px)'></div></div>"
            f"<div class='or-scale-lbl'>{labels}</div>")


def kpi(label: str, value: str, sub: str = "", colour: str = C.PRIMARY, tooltip: str = "", scale: str = "") -> str:
    tip = f" title=\"{tooltip}\"" if tooltip else ""
    return (f"<div class='or-kpi' style='--accent:{colour}'{tip}><div class='l'>{label}</div><div class='v'>{value}</div>"
            f"<div class='s'>{sub}</div>{scale}</div>")


def kpi_row(items: list) -> None:
    """items: list of (label, value, sub, colour, tooltip?, scale_html?)"""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def label(text: str) -> None:
    st.markdown(f"<div class='or-label'>{text}</div>", unsafe_allow_html=True)


def checks(items: list) -> None:
    st.markdown("".join(f"<div class='or-check'>{i}</div>" for i in items), unsafe_allow_html=True)


def page_header(title: str, stage: str | None = None, question: str | None = None, icon: str = "", extra_chips: str = "") -> None:
    chips = (chip(stage.upper(), STAGE_COLOURS.get(stage, C.PRIMARY)) if stage else "") + extra_chips
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin:2px 0 0 0;color:#fff'>{icon} {title}</h1>", unsafe_allow_html=True)
    if question:
        st.markdown(f"<div class='or-problem'><b>Business Problem:</b> {question}</div>", unsafe_allow_html=True)


def card(title: str, body_html: str, colour: str = C.PRIMARY) -> str:
    return f"<div class='or-card'><h4 style='color:{colour} !important'>{title}</h4>{body_html}</div>"


def html_table(df: pd.DataFrame) -> None:
    st.markdown(df.to_html(index=False, classes="or-table", border=0, escape=False), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- streamlit-version-safe wrappers
def _st_version() -> tuple:
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


def _full_width_kwargs(fn=None) -> dict:
    """Streamlit >= 1.50 uses width='stretch' (use_container_width is deprecated); older uses use_container_width=True."""
    return {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}


def show_chart(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, key=key, **_full_width_kwargs(st.plotly_chart))


def show_df(df: pd.DataFrame, height: int | None = None, **kw) -> None:
    # Never pass height=None: newer Streamlit raises StreamlitInvalidHeightError for it.
    if height is not None:
        kw["height"] = int(height)
    st.dataframe(df, hide_index=True, **_full_width_kwargs(st.dataframe), **kw)


def style_fig(fig, height: int = 380, legend_bottom: bool = False):
    """Apply the dark OneRun theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6E6E6", size=12), colorway=C.SEQ, margin=dict(l=10, r=10, t=40, b=10),
        title=dict(font=dict(size=15, color="#fff")),
    )
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=.5))
    fig.update_xaxes(gridcolor="#3a3a3a", zerolinecolor="#3a3a3a")
    fig.update_yaxes(gridcolor="#3a3a3a", zerolinecolor="#3a3a3a")
    return fig


def download_excel(df: pd.DataFrame, label_text: str, filename: str, key: str, sheet: str = "OneRun") -> None:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=sheet[:31], index=False)
    st.download_button(label_text, buf.getvalue(), file_name=filename, key=key,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def inr_cr(v: float) -> str:
    """Format INR crore with Indian-style units."""
    if abs(v) >= 1e5: return f"₹{v / 1e5:,.2f} L Cr"
    return f"₹{v:,.0f} Cr" if abs(v) >= 100 else f"₹{v:,.1f} Cr"
