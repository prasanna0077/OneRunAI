"""Our Bank - profile, KPIs and statistics of the client institution (Meridian Bank)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from onerun import services
from onerun.config import ACQUIRER_ASSUMPTIONS, C
from onerun.ui import theme as th

# (label, column in candidates, higher_is_better)
BENCH = [("ROA", "roa_pct", True), ("ROE", "roe_pct", True), ("CASA ratio", "casa_pct", True),
         ("Gross NPA", "gnpa_pct", False), ("Net NPA", "nnpa_pct", False), ("Provision coverage", "pcr_pct", True),
         ("Capital adequacy", "car_pct", True), ("Net interest margin", "nim_pct", True),
         ("Cost-to-income", "cost_to_income_pct", False), ("Digital adoption", "digital_adoption_pct", True)]


def _benchmark(a: dict, cand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col, hib in BENCH:
        dist = cand[col].astype(float)
        v = float(a[col])
        pct = float((dist < v).mean() * 100 if hib else (dist > v).mean() * 100)   # 100 = better than every candidate
        med = float(dist.median())
        better = (v > med) if hib else (v < med)
        rows.append(dict(Metric=label, Ours=v, Median=med, Percentile=pct, Position="Better" if better else "Weaker"))
    return pd.DataFrame(rows)


def about_text(a: dict, bench: pd.DataFrame, n_candidates: int) -> str:
    b = bench.set_index("Metric")
    strong = [m for m in ("CASA ratio", "Gross NPA", "Digital adoption", "ROA", "Provision coverage")
              if b.loc[m, "Position"] == "Better"]
    weak = [m for m in b.index if b.loc[m, "Position"] == "Weaker"][:3]
    s = (f"**{a['name']}** is a {a['bank_type'].lower()} headquartered in **{a['hq_city']}, {a['hq_state']}** "
         f"(est. {int(a['established'])}). It serves **{a['customers_lakh']:.0f} lakh customers** through "
         f"**{int(a['branches']):,} branches** and about {int(a['employees']):,} employees, with total assets of "
         f"₹{a['total_assets_cr']:,.0f} Cr.")
    s += f" Compared with the {n_candidates} candidate institutions screened, we are ahead on " + (
        ", ".join(strong[:4]) if strong else "few metrics") + "."
    if weak:
        s += f" Areas to watch: {', '.join(weak)}."
    return s


def render() -> None:
    th.page_header("Our Bank", "Across the Lifecycle", None, "🏛")
    a = services.acquirer()
    cand = services.acquisition_results()[1]
    bench = _benchmark(a, cand)

    # ---------------------------------------------------------------- profile header
    st.markdown(th.brand_block(compact=True), unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(f"<h2 style='margin:0;color:#fff !important'>{a['name']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{C.PRIMARY};font-size:15px'>{a['bank_type']} · 📍 {a['hq_city']}, {a['hq_state']} "
                    f"· Est. {int(a['established'])}</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown(about_text(a, bench, len(cand)))
    with right:
        th.kpi_row([
            ("Customers", f"{a['customers_lakh']:.0f} lakh", "", C.PRIMARY),
            ("Branches", f"{int(a['branches']):,}", "", C.DURING),
            ("Employees", f"{int(a['employees']):,}", "", C.AFTER),
        ])
    st.write("")

    # ---------------------------------------------------------------- KPI cards
    th.kpi_row([
        ("Total assets", th.inr_cr(a["total_assets_cr"]), "", C.PRIMARY),
        ("Total deposits", th.inr_cr(a["total_deposits_cr"]), f"CASA {th.inr_cr(a['casa_deposits_cr'])}", C.GREEN),
        ("Total advances", th.inr_cr(a["total_advances_cr"]), f"credit-deposit {a['total_advances_cr'] / a['total_deposits_cr'] * 100:.0f}%", C.BEFORE),
        ("Net worth", th.inr_cr(a["networth_cr"]), f"net profit {th.inr_cr(a['net_profit_cr'])}", C.DURING),
    ])
    st.write("")
    th.kpi_row([
        ("CASA ratio", f"{a['casa_pct']:.1f}%", "of deposits", C.GREEN),
        ("Gross / Net NPA", f"{a['gnpa_pct']:.2f}% / {a['nnpa_pct']:.2f}%", f"{th.inr_cr(a['gross_npa_cr'])} gross", C.RED),
        ("Provision coverage", f"{a['pcr_pct']:.0f}%", f"{th.inr_cr(a['provisions_cr'])} provisions", C.AMBER),
        ("Capital adequacy", f"{a['car_pct']:.1f}%", "CAR", C.PRIMARY),
    ])
    st.write("")
    th.kpi_row([
        ("ROA / ROE", f"{a['roa_pct']:.2f}% / {a['roe_pct']:.1f}%", "", C.GREEN),
        ("Net interest margin", f"{a['nim_pct']:.2f}%", "", C.PRIMARY),
        ("Cost-to-income", f"{a['cost_to_income_pct']:.0f}%", "lower is better", C.AMBER),
        ("Digital adoption", f"{a['digital_adoption_pct']:.0f}%", f"app MAU {a['app_mau_pct']:.0f}% · tech {a['tech_modernity_score']:.1f}/5", C.AFTER),
    ])
    st.write("")

    tab_bs, tab_foot, tab_bench, tab_tbl = st.tabs(["💰 Balance sheet mix", "🗺 Footprint", "📊 Benchmark vs candidates", "📋 All KPIs"])

    # ---------------------------------------------------------------- balance sheet
    with tab_bs:
        c1, c2 = st.columns(2)
        dep = pd.DataFrame({"Type": ["CASA", "Term deposits"], "₹ Cr": [a["casa_deposits_cr"], a["term_deposits_cr"]]})
        p1 = px.pie(dep, names="Type", values="₹ Cr", hole=.5, title="Deposit mix", color="Type",
                    color_discrete_map={"CASA": C.GREEN, "Term deposits": C.PRIMARY})
        p1.update_traces(texttemplate="%{label}<br>%{percent}", textposition="inside")
        with c1:
            th.show_chart(th.style_fig(p1, 360), key="our_dep_pie")
        loans = pd.DataFrame({"Segment": ["Retail", "MSME", "Corporate", "Agri & Gold"],
                              "₹ Cr": [a["loans_retail_cr"], a["loans_msme_cr"], a["loans_corporate_cr"], a["loans_agri_gold_cr"]]})
        p2 = px.pie(loans, names="Segment", values="₹ Cr", hole=.5, title="Loan book mix", color_discrete_sequence=C.SEQ)
        p2.update_traces(texttemplate="%{label}<br>%{percent}", textposition="inside")
        with c2:
            th.show_chart(th.style_fig(p2, 360), key="our_loan_pie")
        aq = pd.DataFrame({"Item": ["Gross NPA", "Provisions held", "Net NPA"],
                           "₹ Cr": [a["gross_npa_cr"], a["provisions_cr"], a["net_npa_cr"]]})
        bar = px.bar(aq, x="Item", y="₹ Cr", text="₹ Cr", color="Item", title="Asset quality (₹ Cr)",
                     color_discrete_map={"Gross NPA": C.RED, "Provisions held": C.GREEN, "Net NPA": C.AMBER})
        bar.update_traces(texttemplate="%{text:,.0f}")
        bar.update_layout(showlegend=False)
        th.show_chart(th.style_fig(bar, 320), key="our_aq_bar")
        st.caption("Rupee amounts are derived from Meridian's ratios using the assumptions in `config.py → ACQUIRER_ASSUMPTIONS`, "
                   "and are stored in the Excel sheet `Acquirer_Financials` (edit that sheet to change them).")

    # ---------------------------------------------------------------- footprint
    with tab_foot:
        share = pd.DataFrame({"Region": list(a["region_branch_share"]),
                              "Branches": [round(v * a["branches"]) for v in a["region_branch_share"].values()],
                              "Share %": [v * 100 for v in a["region_branch_share"].values()]})
        fb = px.bar(share, x="Region", y="Branches", text="Branches", title="Branches by region", color_discrete_sequence=[C.PRIMARY])
        fb.update_traces(texttemplate="%{text:,}")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            th.show_chart(th.style_fig(fb, 340), key="our_branch_bar")
        with c2:
            th.show_df(share.round(1))
            top = share.sort_values("Share %", ascending=False).iloc[0]
            low = share.sort_values("Share %").iloc[:2]
            st.markdown(f"**Insight:** {top.Region} holds {top['Share %']:.0f}% of our branches. "
                        f"{' and '.join(low.Region)} are thinly covered - targets strong there add the most new reach.")

    # ---------------------------------------------------------------- benchmark
    with tab_bench:
        bd = bench.sort_values("Percentile")
        fig = px.bar(bd, x="Percentile", y="Metric", orientation="h", color="Position", text="Percentile",
                     color_discrete_map={"Better": C.GREEN, "Weaker": C.AMBER},
                     title=f"Where we stand among {len(cand)} candidate banks (100 = better than all)",
                     labels={"Percentile": "Percentile (higher = better)", "Metric": ""})
        fig.update_traces(texttemplate="%{text:.0f}")
        fig.add_vline(x=50, line_dash="dot", line_color="#aaa")
        fig.update_xaxes(range=[0, 105])
        th.show_chart(th.style_fig(fig, 420, legend_bottom=True), key="our_bench")
        show = bench.rename(columns={"Ours": "Our Bank", "Median": "Candidate median"}).round(2)
        th.show_df(show)
        better = bench[bench.Position == "Better"].Metric.tolist()
        weaker = bench[bench.Position == "Weaker"].Metric.tolist()
        st.markdown(f"**Insights:** ahead of the candidate median on {', '.join(better) or 'none'}; "
                    f"behind on {', '.join(weaker) or 'none'}. Target banks that are strong where we are weaker create the most complementary value.")

    # ---------------------------------------------------------------- table
    with tab_tbl:
        from onerun.data_generation.acquirer import META
        rows = [(META.get(k, ("", k))[1], v, META.get(k, ("", ""))[0]) for k, v in a.items() if k in META]
        tbl = pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])
        tbl["Value"] = tbl["Value"].astype(str)
        th.show_df(tbl, height=520)
        th.download_excel(tbl, "⬇ Download Our Bank KPIs (Excel)", "OneRun_Our_Bank_KPIs.xlsx", "dl_our_bank", "Our_Bank")
