"""MVP 1 - Acquisition Intelligence (Before the Deal)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from onerun import services
from onerun.config import ACQUIRER, C
from onerun.ml import acquisition as ml
from onerun.modules.base import BaseModule
from onerun.ui import context as ctx
from onerun.ui import theme as th

TIER_COLOURS = {"Prime Target": C.GREEN, "Strong Candidate": C.PRIMARY, "Watchlist": C.AMBER, "Avoid": C.RED}


class AcquisitionModule(BaseModule):
    key = "acquisition"
    title = "Acquisition Intelligence"
    icon = "🎯"
    stage = "Before the Deal"
    phase = 1
    status = "live"
    question = "Which institution should we acquire?"
    engines = "Random Forest, Gradient Boosting, Decision Trees"
    description = "Target identification and acquisition scoring."
    business_value = ["Accelerate target discovery", "Improve acquisition success rate", "Reduce due diligence effort",
                      "Increase probability of synergy realization"]

    def render(self) -> None:
        th.page_header(self.title, self.stage, self.question, self.icon)
        suite, scored = services.acquisition_results()
        cur = ctx.current_target_id()
        cur_row = scored[scored.bank_id == cur].iloc[0]

        th.kpi_row([
            ("Candidates screened", f"{len(scored)}", f"for {ACQUIRER['name']}", C.PRIMARY),
            ("Prime targets", f"{int((scored.tier == 'Prime Target').sum())}", "score ≥ 70", C.GREEN),
            ("Best model (AUC)", f"{suite.metrics.loc[0, 'AUC']:.2f}", suite.best_model, C.BEFORE),
            ("Current deal target", ctx.short_name(cur_row.bank_name), f"score {cur_row.acquisition_score:.0f} · rank #{int(cur_row['rank'])}", C.DURING),
        ])
        st.write("")
        tab_rank, tab_dive, tab_model, tab_data = st.tabs(["🏆 Target ranking", "🔍 Target deep-dive", "🧪 Model lab", "🗂 Data & features"])

        # ------------------------------------------------------------------ ranking
        with tab_rank:
            c1, c2, c3 = st.columns(3)
            types = c1.multiselect("Institution type", sorted(scored.bank_type.unique()), default=sorted(scored.bank_type.unique()))
            regions = c2.multiselect("Region", sorted(scored.region.unique()), default=sorted(scored.region.unique()))
            min_score = c3.slider("Minimum acquisition score", 0, 100, 0, step=5)
            view = scored[scored.bank_type.isin(types) & scored.region.isin(regions) & (scored.acquisition_score >= min_score)]
            if view.empty:
                st.info("No institutions match the filters.")
            else:
                left, right = st.columns([1.1, 1])
                top = view.head(15).iloc[::-1]
                fig = px.bar(top, x="acquisition_score", y="bank_name", color="tier", orientation="h", text="acquisition_score",
                             color_discrete_map=TIER_COLOURS, title="Top candidates by acquisition score",
                             labels={"acquisition_score": "Acquisition score", "bank_name": ""})
                fig.update_traces(texttemplate="%{text:.0f}")
                fig.update_xaxes(range=[0, 105])
                with left:
                    th.show_chart(th.style_fig(fig, 470, legend_bottom=True), key="acq_rank_bar")
                sc = px.scatter(view, x="risk_adj_synergy_cr", y="acquisition_score", size="valuation_cr", color="bank_type",
                                hover_name="bank_name", title="Score vs. risk-adjusted synergy (bubble = valuation)",
                                labels={"risk_adj_synergy_cr": "Risk-adjusted synergy (₹ Cr / yr)", "acquisition_score": "Score"})
                with right:
                    th.show_chart(th.style_fig(sc, 470, legend_bottom=True), key="acq_rank_scatter")
                cols = ["rank", "bank_name", "bank_type", "region", "acquisition_score", "tier", "total_assets_cr", "valuation_cr",
                        "roa_pct", "casa_pct", "gnpa_pct", "pcr_pct", "digital_adoption_pct", "risk_adj_synergy_cr", "red_flags"]
                th.show_df(view[cols], height=420, column_config={
                    "acquisition_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                    "rank": "Rank", "bank_name": "Institution", "bank_type": "Type", "region": "Region", "tier": "Tier",
                    "total_assets_cr": st.column_config.NumberColumn("Assets (₹ Cr)", format="%d"),
                    "valuation_cr": st.column_config.NumberColumn("Valuation (₹ Cr)", format="%d"),
                    "roa_pct": "ROA %", "casa_pct": "CASA %", "gnpa_pct": "GNPA %", "pcr_pct": "PCR %",
                    "digital_adoption_pct": "Digital %", "risk_adj_synergy_cr": st.column_config.NumberColumn("Risk-adj. synergy (₹ Cr)", format="%d"),
                    "red_flags": "Red flags"})
                th.download_excel(view[cols], "⬇ Download ranking (Excel)", "OneRun_Acquisition_Ranking.xlsx", "dl_acq_rank", "Ranking")

        # ------------------------------------------------------------------ deep dive
        with tab_dive:
            labels = ctx.target_labels()
            ids = list(labels)
            sel = st.selectbox("Select an institution", ids, index=ids.index(cur), format_func=labels.get, key="acq_dive_sel")
            r = scored[scored.bank_id == sel].iloc[0]
            if sel == cur:
                st.success("✔ This institution is the **current deal target** - Integration and Retention Intelligence are using it.")
            else:
                st.button("➜ Make this the deal target", on_click=ctx.set_target, args=(sel,), key="acq_set_target")
            th.kpi_row([
                ("Acquisition score", f"{r.acquisition_score:.0f}/100", r.tier, TIER_COLOURS[r.tier]),
                ("Valuation", th.inr_cr(r.valuation_cr), f"{r.price_to_book:.2f}x book", C.PRIMARY),
                ("Indicative synergy", th.inr_cr(r.indicative_synergy_cr), "per year, gross", C.GREEN),
                ("Risk-adjusted synergy", th.inr_cr(r.risk_adj_synergy_cr), f"{r.synergy_to_valuation_pct:.1f}% of valuation", C.BEFORE),
            ])
            st.write("")
            a, b = st.columns([1.1, 1])
            rad = ml.radar_profile(r, scored)
            fig = go.Figure()
            for name, colour in (("Target", C.PRIMARY), ("Acquirer", C.DURING), ("Peer median", "#8a8a8a")):
                fig.add_trace(go.Scatterpolar(r=rad[name].tolist() + [rad[name].iloc[0]], theta=rad.dimension.tolist() + [rad.dimension.iloc[0]],
                                              name=name, line=dict(color=colour, dash="dot" if name == "Peer median" else "solid"),
                                              fill="toself" if name == "Target" else None, opacity=.85))
            fig.update_layout(title=f"{r.bank_name} vs. {ACQUIRER['name']} (percentile among candidates)",
                              polar=dict(radialaxis=dict(range=[0, 100], gridcolor="#444"), bgcolor="rgba(0,0,0,0)", angularaxis=dict(gridcolor="#444")))
            with a:
                th.show_chart(th.style_fig(fig, 430, legend_bottom=True), key="acq_radar")
            with b:
                votes = pd.DataFrame({"Model": list(suite.models), "Probability of synergy realisation":
                                      [r[f"p_{m}"] * 100 for m in suite.models]})
                vf = px.bar(votes, x="Model", y="Probability of synergy realisation", text="Probability of synergy realisation",
                            title="What each model says (%)", color_discrete_sequence=[C.PRIMARY])
                vf.update_traces(texttemplate="%{text:.0f}%")
                vf.update_yaxes(range=[0, 105])
                th.show_chart(th.style_fig(vf, 250), key="acq_votes")
                th.label("Red flags")
                flags = [] if r.red_flags == "None" else r.red_flags.split("; ")
                if flags:
                    st.markdown("".join(f"<div style='color:#ffb3ad;font-size:14px'>⚠ {f}</div>" for f in flags), unsafe_allow_html=True)
                else:
                    st.markdown("<div class='or-check'>No red flags raised on the screening rules</div>", unsafe_allow_html=True)
                th.label("Snapshot")
                st.markdown(f"**{r.bank_type}** · {r.region} · HQ platform **{r.core_banking_platform}** · {int(r.branches):,} branches · "
                            f"{r.customers_lakh:.1f} lakh customers · size = {r.size_ratio * 100:.1f}% of {ACQUIRER['name']}")

        # ------------------------------------------------------------------ model lab
        with tab_model:
            st.caption(f"Trained on **{suite.n_train + suite.n_test:,} historical deals** (label: synergy realised within 36 months - "
                       f"base rate {suite.base_rate * 100:.0f}%). Metrics are on a {suite.n_test}-deal hold-out set.")
            m = suite.metrics.copy()
            for c in ("AUC", "Accuracy", "Precision", "Recall", "F1"):
                m[c] = m[c].round(3)
            th.show_df(m)
            a, b = st.columns(2)
            fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#666", dash="dash"), name="Chance", showlegend=False))
            for name, (fpr, tpr, auc) in suite.roc.items():
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.2f})"))
            fig.update_layout(title="ROC curves", xaxis_title="False-positive rate", yaxis_title="True-positive rate")
            with a:
                th.show_chart(th.style_fig(fig, 400, legend_bottom=True), key="acq_roc")
            imp = suite.importance.mean(axis=1).sort_values(ascending=False).head(12)
            imp.index = [ml.FEATURE_LABELS.get(i, i) for i in imp.index]
            fi = px.bar(imp.iloc[::-1], orientation="h", title="What drives acquisition success (avg. importance)",
                        labels={"value": "Importance", "index": ""}, color_discrete_sequence=[C.GREEN])
            fi.update_layout(showlegend=False)
            with b:
                th.show_chart(th.style_fig(fi, 400), key="acq_imp")

        # ------------------------------------------------------------------ data
        with tab_data:
            raw = services.acquisition_data()
            th.label("Datasets used (Excel: data/OneRun_Acquisition_Data.xlsx)")
            th.html_table(pd.DataFrame({
                "Data source": ["Bank Financials", "Credit Metrics", "Business Metrics", "Digital Metrics"],
                "Features": ["ROA, ROE, CASA, NIM, cost-to-income, valuation", "GNPA, NNPA, PCR, CAR", "Branches, customers, size & geographic fit",
                             "App usage, digital adoption, tech modernity"]}))
            st.write("")
            st.markdown("**Candidate institutions (merged view)**")
            th.show_df(raw["candidates"], height=260)
            st.markdown("**Historical deals used for training**")
            th.show_df(raw["history"].head(200), height=260)
