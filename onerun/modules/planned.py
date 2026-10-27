"""Roadmap modules: the 11 intelligence layers outside the MVP.

* Executive Intelligence is a working *foundation* (template narrative over the three MVPs' outputs).
* All others are placeholders that already declare their question, engine and the MVP data they will reuse,
  so the navigation, roadmap and architecture pages show them today.
To activate one: subclass BaseModule, implement render(), set status = "live", and swap it in registry.py.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from onerun import services
from onerun.ai.narrative import get_engine, recommendation, verdict_rules
from onerun.config import ASSUMPTIONS, THRESHOLDS, C
from onerun.ml import proforma as pf
from onerun.modules.base import BaseModule
from onerun.ui import context as ctx
from onerun.ui import theme as th


class PlannedModule(BaseModule):
    status = "planned"
    reuses: str = ""          # which MVP outputs / datasets this module will build on
    new_data: str = ""        # extra datasets needed

    def __init__(self, key, title, icon, stage, phase, question, engines, reuses="", new_data=""):
        self.key, self.title, self.icon, self.stage, self.phase = key, title, icon, stage, phase
        self.question, self.engines, self.reuses, self.new_data = question, engines, reuses, new_data

    def render(self) -> None:
        th.page_header(self.title, self.stage, self.question, self.icon,
                       extra_chips=th.chip(f"PHASE {self.phase}", C.AMBER))
        st.info(f"🚧 **Planned for Phase {self.phase}.** The platform foundation is ready - this module plugs into it without changes elsewhere.")
        a, b, c = st.columns(3)
        a.markdown(th.card("AI / ML engine", f"<p>{self.engines}</p>"), unsafe_allow_html=True)
        b.markdown(th.card("Builds on", f"<p>{self.reuses or 'Deal Context'}</p>"), unsafe_allow_html=True)
        c.markdown(th.card("New data needed", f"<p>{self.new_data or 'None - uses existing datasets'}</p>"), unsafe_allow_html=True)
        st.write("")
        st.markdown("**How to activate** - create `onerun/modules/<name>_module.py` with a `BaseModule` subclass, "
                    "read inputs via `services.deal_context(bank_id)`, then register it in `onerun/modules/registry.py`.")


class ExecutiveModule(BaseModule):
    key, title, icon = "executive", "Executive Intelligence", "📊"
    stage, phase, status = "Across the Lifecycle", 2, "foundation"
    question = "What is the overall merger outlook?"
    engines = "GenAI + Insights Layer"

    def render(self) -> None:
        th.page_header(self.title, self.stage, self.question, self.icon, extra_chips=th.chip("PHASE 2 · FOUNDATION", C.AMBER))
        bank_id = ctx.target_selector("exec")
        with st.spinner("Assembling deal context from the three MVPs ..."):
            dc = services.deal_context(bank_id)
        k = dc["kpis"]
        verdict, _ = recommendation(k)
        colour = {"PROCEED": C.GREEN, "PROCEED WITH CONDITIONS": C.AMBER, "HOLD": C.AMBER, "DO NOT PROCEED": C.RED}[verdict]
        A, T = ASSUMPTIONS, THRESHOLDS
        acq_bands = [(A["tier_watch"], C.RED, f"Avoid <{A['tier_watch']}"), (A["tier_strong"], C.AMBER, "Watch"),
                     (A["tier_prime"], C.PRIMARY, "Strong"), (100, C.GREEN, f"Prime ≥{A['tier_prime']}")]
        cx_bands = [(T["complexity_low"], C.GREEN, f"Low <{T['complexity_low']}"), (T["complexity_high"], C.AMBER, "Moderate"),
                    (100, C.RED, f"High ≥{T['complexity_high']}")]
        ch_bands = [(T["churn_reference_pct"], C.GREEN, f"<{T['churn_reference_pct']:.0f}% typical"), (T["churn_limit_pct"], C.AMBER, "watch"),
                    (40, C.RED, f">{T['churn_limit_pct']:.0f}% limit")]
        th.kpi_row([
            ("Verdict", verdict.title(), dc["target"].bank_name, colour,
             "Decided by three tests - see 'Why this verdict' below."),
            ("Acquisition score", f"{k['acquisition_score']:.0f}/100", k["tier"], C.BEFORE,
             "Model-estimated probability (0-100) that the deal realises its synergies.",
             th.scale_bar(k["acquisition_score"], acq_bands, 0, 100)),
            ("Integration complexity", f"{k['complexity_score']:.0f}/100", k["complexity_band"], C.DURING,
             "Weighted effort of migrating customers, branches, products and the core banking system.",
             th.scale_bar(k["complexity_score"], cx_bands, 0, 100)),
            ("Expected churn", f"{k['expected_churn_pct']:.1f}%", f"{k['high_risk_customers']} high-risk customers", C.AFTER,
             "Share of customers expected to leave within 12 months of the merger. Dashed line = average of past mergers.",
             th.scale_bar(k["expected_churn_pct"], ch_bands, 0, 40, ref=T["churn_reference_pct"])),
        ])
        st.write("")

        with st.expander("ℹ How to read these indicators (thresholds)", expanded=True):
            def sw(c): return f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:{c}'></span>"
            legend = pd.DataFrame({
                "Indicator": ["Acquisition score", "", "", "", "Integration complexity", "", "", "Expected churn", "", ""],
                "Band": [f"{sw(C.GREEN)} Prime Target", f"{sw(C.PRIMARY)} Strong Candidate", f"{sw(C.AMBER)} Watchlist", f"{sw(C.RED)} Avoid",
                         f"{sw(C.GREEN)} Low", f"{sw(C.AMBER)} Moderate", f"{sw(C.RED)} High",
                         f"{sw(C.GREEN)} Below past-merger average", f"{sw(C.AMBER)} Above average, within limit", f"{sw(C.RED)} Above the limit"],
                "Range": [f"≥ {A['tier_prime']}", f"{A['tier_strong']} - {A['tier_prime'] - 1}", f"{A['tier_watch']} - {A['tier_strong'] - 1}",
                          f"< {A['tier_watch']}", f"< {T['complexity_low']}", f"{T['complexity_low']} - {T['complexity_high'] - 1}",
                          f"≥ {T['complexity_high']}", f"< {T['churn_reference_pct']:.0f}%",
                          f"{T['churn_reference_pct']:.0f}% - {T['churn_limit_pct']:.0f}%", f"≥ {T['churn_limit_pct']:.0f}%"],
                "Meaning": ["Pursue - high chance of realising synergies", "Worth deeper diligence", "Mixed fundamentals - hold", "Low chance - do not pursue",
                            "Straightforward integration", "Needs a plan for waves, branches and products", "Major programme - high execution risk",
                            "Customer base looks stable", "Retention budget advised", "Verdict cannot be a clean 'Proceed'"],
            })
            th.html_table(legend)
            st.caption("Thresholds live in `onerun/config.py` (ASSUMPTIONS and THRESHOLDS) - change them there and this legend, the bars and the verdict update together. "
                       "The white marker is this deal; the dashed line is the historical average churn.")

        th.label("Why this verdict")
        for text, ok in verdict_rules(k):
            st.markdown(f"<div class='or-rule'><span class='{'ok' if ok else 'no'}'>{'✔' if ok else '✖'}</span> {text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='or-rule'><b>→ {verdict.title()}</b> - all three pass = Proceed; Prime or Strong tier but a failed test = Proceed with conditions; "
                    "Watchlist = Hold; Avoid = Do not proceed.</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown(get_engine().generate(dc))
        st.caption("Narrative is template-generated today. `onerun/ai/narrative.py` is the hook for a GenAI engine in Phase 3.")

        self._combined_bank(dc)

    # ------------------------------------------------------------------ post-merger view
    def _combined_bank(self, dc: dict) -> None:
        res, t = dc["combined"], dc["target"]
        base, tab = res["base"], res["table"]
        st.markdown("---")
        st.markdown(f"## 🏦 Post-merger: combined bank")
        st.caption(f"Day-1 simple merge of **{dc['acquirer']}** + **{t.bank_name}**: balance sheets are added, ratios are recalculated from the "
                   "totals. No synergies, customer run-off or purchase-accounting adjustments are included.")
        v = tab.set_index("key")
        th.kpi_row([
            ("Combined assets", th.inr_cr(v.loc["total_assets_cr", "Combined"]), f"+{v.loc['total_assets_cr', 'Change %']:.0f}% vs today", C.PRIMARY),
            ("Combined deposits", th.inr_cr(v.loc["total_deposits_cr", "Combined"]), f"CASA {v.loc['casa_pct', 'Combined']:.1f}%", C.GREEN),
            ("Combined loans", th.inr_cr(v.loc["total_advances_cr", "Combined"]), f"gross NPA {v.loc['gnpa_pct', 'Combined']:.2f}%", C.RED),
            ("Customers / branches", f"{v.loc['customers_lakh', 'Combined']:.0f} L / {v.loc['branches', 'Combined']:,.0f}",
             f"from {v.loc['customers_lakh', 'Our Bank']:.0f} L / {v.loc['branches', 'Our Bank']:,.0f}", C.DURING),
        ])
        st.write("")
        banks = ["Our Bank", "Target", "Combined"]
        colours = {"Our Bank": C.PRIMARY, "Target": C.BEFORE, "Combined": C.GREEN}

        def grouped(keys, title, key, fmt):
            d = tab[tab.key.isin(keys)].melt(id_vars="Metric", value_vars=banks, var_name="Bank", value_name="Value")
            f = px.bar(d, x="Metric", y="Value", color="Bank", barmode="group", title=title, color_discrete_map=colours,
                       category_orders={"Bank": banks, "Metric": [METRICS_LABEL[k] for k in keys]})
            f.update_traces(texttemplate=fmt, textposition="outside")
            th.show_chart(th.style_fig(f, 380, legend_bottom=True), key=key)

        c1, c2 = st.columns(2)
        with c1:
            grouped(["total_assets_cr", "total_deposits_cr", "total_advances_cr", "networth_cr"], "Balance sheet (₹ Cr)", "exec_bs", "%{y:,.0f}")
        with c2:
            grouped(["gross_npa_cr", "provisions_cr", "net_npa_cr", "net_profit_cr"], "Asset quality and profit (₹ Cr)", "exec_aq", "%{y:,.0f}")
        c1, c2 = st.columns(2)
        with c1:
            grouped(["casa_pct", "pcr_pct", "car_pct", "digital_adoption_pct"], "Franchise strength ratios (%) - higher is better", "exec_r1", "%{y:.1f}")
        with c2:
            grouped(["gnpa_pct", "nnpa_pct", "roa_pct", "roe_pct", "nim_pct", "cost_to_income_pct"], "Quality and profitability ratios (%)", "exec_r2", "%{y:.2f}")
        c1, c2 = st.columns(2)
        dep = pf.deposit_split(base["Combined"])
        pie = px.pie(dep, names="Type", values="₹ Cr", hole=.5, title="Combined deposit mix", color="Type",
                     color_discrete_map={"CASA": C.GREEN, "Term deposits": C.PRIMARY})
        pie.update_traces(texttemplate="%{label}<br>%{percent}", textposition="inside")
        with c1:
            th.show_chart(th.style_fig(pie, 340), key="exec_dep_pie")
        sizes = pd.DataFrame({"Bank": banks, "Assets": [base[b]["total_assets_cr"] for b in banks]})
        sp = px.pie(sizes[sizes.Bank != "Combined"], names="Bank", values="Assets", hole=.5, title="Who contributes the assets",
                    color="Bank", color_discrete_map=colours)
        sp.update_traces(texttemplate="%{label}<br>%{percent}", textposition="inside")
        with c2:
            th.show_chart(th.style_fig(sp, 340), key="exec_contrib_pie")

        def fmt(row, col):
            x = row[col]
            return f"{x:,.0f}" if row.Unit in ("₹ Cr", "#") else (f"{x:,.1f}" if row.Unit == "lakh" else f"{x:.2f}")
        show = pd.DataFrame({
            "Metric": tab.Metric, "Unit": tab.Unit,
            "Our Bank": [fmt(r, "Our Bank") for _, r in tab.iterrows()], "Target": [fmt(r, "Target") for _, r in tab.iterrows()],
            "Combined": [fmt(r, "Combined") for _, r in tab.iterrows()],
            "Change vs today": [f"{r.Change:+,.2f}" if r.Unit == "%" else f"{r.Change:+,.0f}" for _, r in tab.iterrows()],
            "Impact": tab.Impact.map({"Better": "✅ Better", "Worse": "⚠ Worse", "Neutral": "• Neutral", "No change": "• No change"}),
        })
        th.show_df(show, height=420)
        th.label("What it means")
        st.markdown("\n".join(f"- {line}" for line in pf.summary_lines(res, dc["acquirer"], t.bank_name)))
        th.download_excel(tab.drop(columns="key"), "⬇ Download combined-bank table (Excel)", "OneRun_Combined_Bank.xlsx", "dl_combined", "Combined_Bank")


METRICS_LABEL = {k: v[0] for k, v in pf.METRICS.items()}


def planned_modules() -> list:
    P = PlannedModule
    return [
        P("risk_dd", "Risk & Due Diligence Intelligence", "🔬", "Before the Deal", 2, "What hidden risks exist in the target?",
          "Logistic Regression, Fraud Detection Models", "Acquisition candidates, Credit_Metrics", "Loan-level / fraud-event data"),
        P("financial", "Financial Intelligence", "💹", "Before the Deal", 4, "Is the target financially strong?",
          "Classification Models", "Bank_Financials", "Multi-year financial statements"),
        P("segmentation", "Customer Segmentation Intelligence", "🧩", "During the Deal", 4, "What customer segments exist?",
          "K-Means Clustering", "Already delivered inside Integration Intelligence (MVP)", ""),
        P("synergy", "Synergy Intelligence", "🔗", "During the Deal", 2, "How much value can the merger create?",
          "Regression / Boosting Models", "Indicative synergy, branch savings, product plan", "Realised-synergy tracking"),
        P("branch_opt", "Branch Optimization Intelligence", "🏦", "During the Deal", 4, "Which branches should be retained, merged or shut down?",
          "Clustering", "Delivered as a tab inside Integration Intelligence", "Catchment / demographic data"),
        P("product_rat", "Product Rationalization Intelligence", "📦", "During the Deal", 4, "Which products should survive post-merger?",
          "Clustering + Analytics", "Delivered as a tab inside Integration Intelligence", ""),
        P("tech_int", "Technology Integration Intelligence", "🖥", "During the Deal", 4, "Which systems should be retained or retired?",
          "Clustering + Scoring Models", "core_banking_platform, tech_modernity_score", "Application inventory"),
        P("workforce", "Workforce Intelligence", "👷", "During the Deal", 4, "Which teams overlap?",
          "Clustering", "Branch staff_count, employees", "HR master data"),
        P("cross_sell", "Customer Cross-Sell Intelligence", "🛍", "After the Deal", 4, "Which products should be offered after integration?",
          "Classification Models", "Customers, Products", "Product-propensity labels"),
        P("churn_prev", "Churn Prevention Intelligence", "🛟", "After the Deal", 4, "How can churn be prevented?",
          "Predictive Models + Recommendations", "Delivered inside Retention Intelligence (playbook + budget simulator)", ""),
        P("genai", "GenAI Deal Assistant", "💬", "Across the Lifecycle", 3, "Ask anything about the deal.",
          "LLM + retrieval over Deal Context", "services.deal_context()", "LLM API key"),
        P("regulatory", "Regulatory Intelligence", "⚖️", "Across the Lifecycle", 3, "Are we compliant with RBI / CCI requirements?",
          "Rules engine + document AI", "Acquisition red flags", "Regulatory circulars"),
        P("board", "Automated Board Reporting", "📑", "Across the Lifecycle", 3, "Generate the board pack.",
          "GenAI + report templates", "Executive Intelligence narrative", "Board templates"),
    ]
