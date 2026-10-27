"""MVP 3 - Customer Retention Intelligence (After the Deal)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from onerun import services
from onerun.config import ASSUMPTIONS, C
from onerun.ml import retention as ml
from onerun.modules.base import BaseModule
from onerun.ui import context as ctx
from onerun.ui import theme as th

BAND_COLOURS = {"Low": C.GREEN, "Medium": C.AMBER, "High": C.RED}


class RetentionModule(BaseModule):
    key = "retention"
    title = "Customer Retention Intelligence"
    icon = "🛡"
    stage = "After the Deal"
    phase = 1
    status = "live"
    question = "Which customers will leave after the merger?"
    engines = "Logistic Regression, Random Forest, Gradient Boosting"
    description = "Churn prediction and retention strategies."
    business_value = ["Predict customer attrition", "Reduce revenue leakage", "Improve customer retention", "Protect merger value"]

    def render(self) -> None:
        th.page_header(self.title, self.stage, self.question, self.icon)
        bank_id = ctx.target_selector("ret")
        t = services.target_row(bank_id)
        suite = services.retention_suite()

        c1, c2, c3 = st.columns([1.2, 1, 1])
        model = c1.selectbox("Scoring model", list(suite.models), index=list(suite.models).index(suite.best_model),
                             help="Defaults to the model with the best hold-out AUC.")
        med = c2.slider("Medium-risk threshold", .05, .40, float(ASSUMPTIONS["risk_medium"]), .01)
        high = c3.slider("High-risk threshold", .20, .80, float(ASSUMPTIONS["risk_high"]), .01)
        high = max(high, med + .01)
        with st.spinner("Scoring customers ..."):
            sc = services.retention_scored(bank_id, model, med, high)
            integ = services.integration_results(bank_id)
        scale = integ.scale_customers
        rar_cr = sc.revenue_at_risk_inr.sum() / 1e7 * scale
        n_high = int((sc.risk_band == "High").sum())
        th.kpi_row([
            ("Customers scored", f"{len(sc):,}", f"sample of {t.customers_lakh:.1f} lakh", C.PRIMARY),
            ("Expected churn", f"{sc.churn_probability.mean() * 100:.1f}%", "12 months post-merger", C.AFTER),
            ("High-risk customers", f"{n_high}", f"{n_high / len(sc) * 100:.0f}% of base", C.RED),
            ("Revenue at risk", th.inr_cr(rar_cr), "annual, projected to full base", C.AMBER),
        ])
        st.write("")
        tabs = st.tabs(["🚨 Churn risk", "🧠 Drivers & models", "🛟 Retention playbook", "🎛 Budget simulator", "🔎 Customer lens"])

        # ------------------------------------------------------------------ risk
        with tabs[0]:
            a, b = st.columns(2)
            hist = px.histogram(sc, x="churn_probability", color="risk_band", nbins=30, color_discrete_map=BAND_COLOURS,
                                category_orders={"risk_band": ["Low", "Medium", "High"]}, title="Churn probability distribution",
                                labels={"churn_probability": "Predicted churn probability"})
            with a:
                th.show_chart(th.style_fig(hist, 360, legend_bottom=True), key="ret_hist")
            seg = (sc.groupby("segment").agg(avg=("churn_probability", "mean"), n=("customer_id", "count")).reset_index()
                   .sort_values("avg", ascending=False))
            seg["avg"] = seg.avg * 100
            sb = px.bar(seg, x="segment", y="avg", text="avg", title="Average churn risk by integration segment (%)",
                        labels={"avg": "Avg churn probability %", "segment": ""}, color_discrete_sequence=[C.AFTER])
            sb.update_traces(texttemplate="%{text:.0f}%")
            with b:
                th.show_chart(th.style_fig(sb, 360), key="ret_seg")
            exp = (sc.groupby(["branch_closure_flag", "product_sunset_flag"]).agg(n=("customer_id", "count"), churn=("churn_probability", "mean")).reset_index())
            exp["Integration exposure"] = exp.apply(lambda r: ("Branch consolidated" if r.branch_closure_flag else "Branch stable") + " · " +
                                                    ("Product sunset" if r.product_sunset_flag else "Product retained"), axis=1)
            exp["Avg churn %"] = (exp.churn * 100).round(1)
            st.markdown("**How Integration decisions move churn** (hand-off from Integration Intelligence)")
            th.html_table(exp[["Integration exposure", "n", "Avg churn %"]].rename(columns={"n": "Customers"}).sort_values("Avg churn %", ascending=False))
            st.write("")
            top = sc.sort_values("revenue_at_risk_inr", ascending=False).head(50)
            cols = ["customer_id", "segment", "home_city", "age", "tenure_years", "avg_balance_inr", "churn_probability", "risk_band",
                    "revenue_at_risk_inr", "top_reasons", "recommended_action"]
            st.markdown("**Top 50 customers by revenue at risk**")
            th.show_df(top[cols], height=360, column_config={
                "churn_probability": st.column_config.ProgressColumn("Churn prob.", min_value=0, max_value=1, format="%.2f")})
            th.download_excel(sc[cols + ["channel", "expected_saving_inr", "net_value_inr"]], "⬇ Download scored customers (Excel)",
                              f"OneRun_Churn_Scores_{bank_id}.xlsx", "dl_ret_scores", "Churn_Scores")

        # ------------------------------------------------------------------ drivers
        with tabs[1]:
            st.caption(f"Trained on **{suite.n_train + suite.n_test:,} customers from 6 completed mergers** "
                       f"(12-month churn base rate {suite.base_rate * 100:.0f}%). Threshold for precision/recall: 0.35.")
            m = suite.metrics.copy()
            for c in ("AUC", "Accuracy", "Precision", "Recall", "F1"):
                m[c] = m[c].round(3)
            th.show_df(m)
            a, b = st.columns(2)
            fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#666", dash="dash"), showlegend=False))
            for name, (fpr, tpr, auc) in suite.roc.items():
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.2f})"))
            fig.update_layout(title="ROC curves", xaxis_title="False-positive rate", yaxis_title="True-positive rate")
            with a:
                th.show_chart(th.style_fig(fig, 400, legend_bottom=True), key="ret_roc")
            imp = suite.importance[model].sort_values(ascending=False).head(12)
            imp.index = [ml.FEATURE_LABELS.get(i, i) for i in imp.index]
            fi = px.bar(imp.iloc[::-1], orientation="h", title=f"Churn drivers - {model}", labels={"value": "Importance", "index": ""},
                        color_discrete_sequence=[C.AFTER])
            fi.update_layout(showlegend=False)
            with b:
                th.show_chart(th.style_fig(fi, 400), key="ret_imp")

        # ------------------------------------------------------------------ playbook
        with tabs[2]:
            rc = ml.reason_counts(sc)
            fig = px.bar(rc.sort_values("revenue_at_risk_cr"), x="revenue_at_risk_cr", y="reason", orientation="h",
                         title="Revenue at risk by churn reason (₹ Cr, sample - customers can have several reasons)",
                         labels={"revenue_at_risk_cr": "₹ Cr (sample)", "reason": ""}, color_discrete_sequence=[C.AFTER])
            th.show_chart(th.style_fig(fig, 380), key="ret_reasons")
            th.show_df(rc.rename(columns={"reason": "Churn reason", "customers": "Customers", "avg_churn_probability": "Avg churn prob.",
                                          "revenue_at_risk_cr": "Revenue at risk (₹ Cr, sample)", "action": "Retention strategy",
                                          "channel": "Channel", "cost_per_customer_inr": "Cost / customer (₹)",
                                          "expected_uplift_pct": "Expected churn reduction %"}).round(2))
            st.caption("Cost and uplift figures in the playbook are planning assumptions (edit them in `onerun/ml/retention.py → PLAYBOOK`).")

        # ------------------------------------------------------------------ budget simulator
        with tabs[3]:
            elig = sc[(sc.net_value_inr > 0) & (sc.action_cost_inr > 0)]
            max_k = max(int(elig.action_cost_inr.sum() / 1000) + 1, 20)
            budget = st.slider("Retention budget (₹ thousand, sample base)", 5, max_k, max(10, max_k // 2), 5)
            sim = ml.simulate_budget(sc, budget * 1000)
            th.kpi_row([
                ("Customers targeted", f"{sim['n']:,}", f"of {sim['eligible']:,} eligible", C.PRIMARY),
                ("Spend", f"₹{sim['spend'] / 1e3:,.0f} K", "", C.AMBER),
                ("Revenue protected", f"₹{sim['saved'] / 1e3:,.0f} K", "expected, annual", C.GREEN),
                ("Return on spend", f"{sim['roi']:.1f}x", f"net ₹{sim['net'] / 1e3:,.0f} K", C.AFTER),
            ])
            st.write("")
            step = max(5, max_k // 20)
            sweep = pd.DataFrame([dict(budget=b, net=(lambda s: (s["saved"] - s["spend"]) / 1e3)(ml.simulate_budget(sc, b * 1000)))
                                  for b in range(step, max_k + step, step)])
            line = px.line(sweep, x="budget", y="net", markers=True, title="Net value vs. retention budget (diminishing returns)",
                           labels={"budget": "Budget (₹ thousand)", "net": "Net value (₹ K)"}, color_discrete_sequence=[C.GREEN])
            th.show_chart(th.style_fig(line, 340), key="ret_sweep")
            if sim["n"]:
                th.show_df(sim["targeted"][["customer_id", "segment", "churn_probability", "lead_reason", "recommended_action", "channel",
                                            "action_cost_inr", "expected_saving_inr"]].head(100), height=300)

        # ------------------------------------------------------------------ customer lens
        with tabs[4]:
            pick = st.selectbox("Customer", sc.sort_values("churn_probability", ascending=False).customer_id.head(200).tolist())
            r = sc[sc.customer_id == pick].iloc[0]
            a, b = st.columns([1, 1.4])
            g = go.Figure(go.Indicator(mode="gauge+number", value=r.churn_probability * 100, number=dict(suffix="%"),
                                       title=dict(text=f"Churn probability · {r.risk_band} risk"),
                                       gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=BAND_COLOURS[r.risk_band]),
                                                  steps=[dict(range=[0, med * 100], color="#1f4d33"), dict(range=[med * 100, high * 100], color="#5a4a14"),
                                                         dict(range=[high * 100, 100], color="#5a1f1c")])))
            with a:
                th.show_chart(th.style_fig(g, 320), key="ret_lens_gauge")
            with b:
                st.markdown(f"**{r.customer_id}** · {r.segment} · {r.home_city} · age {int(r.age)} · tenure {r.tenure_years:.1f} yrs · "
                            f"{int(r.products_held)} products · balance ₹{r.avg_balance_inr:,.0f}")
                th.label("Why they may leave")
                st.write(r.top_reasons)
                th.label("Recommended action")
                st.write(f"{r.recommended_action}  ·  *via {r.channel}*")
                st.caption(f"Cost ₹{r.action_cost_inr:,.0f} · expected revenue saved ₹{r.expected_saving_inr:,.0f} · net ₹{r.net_value_inr:,.0f}")
