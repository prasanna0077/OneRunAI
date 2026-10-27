"""MVP 2 - Integration Intelligence (During the Deal)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from onerun import services
from onerun.config import C
from onerun.ml import integration as ml
from onerun.modules.base import BaseModule
from onerun.ui import context as ctx
from onerun.ui import theme as th

ACTION_COLOURS = {"Retain": C.GREEN, "Retain - monitor": C.AMBER, "Merge into acquirer branch": C.DURING, "Shut down": C.RED}
PRODUCT_COLOURS = {"Retain & cross-sell": C.GREEN, "Harmonise with acquirer product": C.PRIMARY,
                   "Sunset & migrate customers": C.RED, "Review selectively": C.AMBER}


class IntegrationModule(BaseModule):
    key = "integration"
    title = "Integration Intelligence"
    icon = "🧬"
    stage = "During the Deal"
    phase = 1
    status = "live"
    question = "How do we integrate customers, products and operations?"
    engines = "K-Means, Hierarchical Clustering"
    description = "Customer segmentation and integration planning."
    business_value = ["Reduce integration complexity", "Rationalize products", "Optimize customer migration", "Accelerate synergy capture"]

    def render(self) -> None:
        th.page_header(self.title, self.stage, self.question, self.icon)
        bank_id = ctx.target_selector("int")
        t = services.target_row(bank_id)

        with st.expander("⚙ Clustering settings", expanded=False):
            c1, c2, c3 = st.columns(3)
            kc = c1.slider("Customer segments (K-Means)", 3, 7, 5)
            kp = c2.slider("Product clusters (Hierarchical)", 2, 6, 4)
            kb = c3.slider("Branch tiers (K-Means)", 2, 4, 3)
        with st.spinner("Running clustering models ..."):
            res = services.integration_results(bank_id, kc, kp, kb)
        cx = res.complexity
        band_col = {"Low": C.GREEN, "Moderate": C.AMBER, "High": C.RED}[cx["band"]]
        saving_cr = res.branches.annual_saving_lakh.sum() / 100 * res.scale_branches
        th.kpi_row([
            ("Customers analysed", f"{len(res.customers):,}", f"sample of {t.customers_lakh:.1f} lakh", C.PRIMARY),
            ("Integration complexity", f"{cx['score']:.0f}/100", f"{cx['band']}", band_col),
            ("Branch cost synergy", th.inr_cr(saving_cr), "annual, projected to full network", C.GREEN),
            ("Products to sunset", f"{int(res.products.product_sunset_flag.sum())}/{len(res.products)}", "redundant vs. acquirer", C.DURING),
        ])
        st.write("")
        tabs = st.tabs(["👥 Customer segments", "📦 Product rationalisation", "🏦 Branch consolidation", "🗺 Integration plan"])

        # ------------------------------------------------------------------ customers
        with tabs[0]:
            a, b = st.columns([1.4, 1])
            seg = res.segments
            fig = px.scatter(res.customers, x="login_frequency_30d", y="avg_balance_inr", color="segment", log_y=True,
                             opacity=.7, hover_data=["age", "tenure_years", "products_held"],
                             title="Customer segments: digital engagement vs. balance (log scale)",
                             labels={"login_frequency_30d": "Digital log-ins (30d)", "avg_balance_inr": "Avg balance (₹)"})
            with a:
                th.show_chart(th.style_fig(fig, 420, legend_bottom=True), key="int_seg_scatter")
            sil = px.line(res.silhouette, x="k", y="silhouette", markers=True, title="Silhouette score by number of segments",
                          labels={"k": "Segments (k)", "silhouette": "Silhouette"})
            sil.add_vline(x=kc, line_dash="dot", line_color=C.PRIMARY)
            with b:
                th.show_chart(th.style_fig(sil, 420), key="int_sil")
            show = seg.rename(columns={"segment": "Segment", "customers": "Customers", "share_pct": "Share %", "avg_age": "Avg age",
                                       "avg_tenure": "Tenure (yrs)", "avg_balance_inr": "Avg balance (₹)", "avg_products": "Products",
                                       "avg_logins_30d": "Log-ins 30d", "avg_branch_visits": "Branch visits", "revenue_cr": "Revenue (₹ Cr, sample)",
                                       "wave": "Wave", "approach": "Migration approach"}).round(1)
            th.show_df(show)

        # ------------------------------------------------------------------ products
        with tabs[1]:
            pr = res.products
            a, b = st.columns([1.3, 1])
            fig = px.scatter(pr, x="feature_overlap_pct", y="annual_revenue_cr", size="active_accounts", color="recommended_action",
                             hover_name="product_name", color_discrete_map=PRODUCT_COLOURS, log_y=True,
                             title="Products: overlap with acquirer vs. revenue (bubble = accounts)",
                             labels={"feature_overlap_pct": "Feature overlap with acquirer product (%)", "annual_revenue_cr": "Revenue (₹ Cr, log)"})
            with a:
                th.show_chart(th.style_fig(fig, 420, legend_bottom=True), key="int_prod_scatter")
            counts = pr.recommended_action.value_counts().reset_index()
            counts.columns = ["action", "products"]
            bar = px.bar(counts, x="products", y="action", orientation="h", color="action", color_discrete_map=PRODUCT_COLOURS,
                         title="Recommended product actions", labels={"action": ""})
            bar.update_layout(showlegend=False)
            with b:
                th.show_chart(th.style_fig(bar, 420), key="int_prod_bar")
            if len(pr) > 3:
                try:
                    import plotly.figure_factory as ff
                    dend = ff.create_dendrogram(ml.product_matrix(pr), labels=pr.product_name.tolist(), orientation="left")
                    dend.update_layout(title="Hierarchical clustering of products (Ward linkage)")
                    th.show_chart(th.style_fig(dend, max(380, 20 * len(pr))), key="int_dend")
                except Exception:       # scipy / figure_factory unavailable -> skip silently
                    pass
            th.show_df(pr[["product_id", "product_name", "product_type", "active_accounts", "annual_revenue_cr", "feature_overlap_pct",
                           "acquirer_equivalent", "cluster", "recommended_action"]], height=320)

        # ------------------------------------------------------------------ branches
        with tabs[2]:
            br = res.branches
            vc = br.recommended_action.value_counts()
            th.kpi_row([
                ("Branches in sample", f"{len(br)}", f"of {int(t.branches):,} in network", C.PRIMARY),
                ("Merge into acquirer", f"{int(vc.get('Merge into acquirer branch', 0))}", "within ~3 km of an acquirer branch", C.DURING),
                ("Shut down", f"{int(vc.get('Shut down', 0))}", "low productivity, no nearby alternative", C.RED),
                ("Annual saving (sample)", f"₹{br.annual_saving_lakh.sum():,.0f} L", "rent + staff + opex", C.GREEN),
            ])
            st.write("")
            geo = px.scatter_geo(br, lat="lat", lon="lon", color="recommended_action", size="footfall_daily", hover_name="city",
                                 hover_data=["branch_id", "branch_tier", "monthly_cost_lakh"], color_discrete_map=ACTION_COLOURS,
                                 title="Branch footprint and recommended action")
            geo.update_geos(scope="asia", center=dict(lat=22, lon=80), projection_scale=4.2, showcountries=True, countrycolor="#555",
                            landcolor="#2e2e2e", bgcolor="rgba(0,0,0,0)", showocean=False, lataxis_range=[6, 36], lonaxis_range=[67, 98])
            th.show_chart(th.style_fig(geo, 520, legend_bottom=True), key="int_geo")
            tier = px.box(br, x="branch_tier", y="cost_per_txn_inr", color="branch_tier", title="Cost per transaction by branch tier (₹)",
                          labels={"branch_tier": "", "cost_per_txn_inr": "₹ per transaction"})
            tier.update_layout(showlegend=False)
            a, b = st.columns([1, 1.3])
            with a:
                th.show_chart(th.style_fig(tier, 360), key="int_tier")
            with b:
                th.show_df(br[["branch_id", "city", "branch_type", "footfall_daily", "monthly_cost_lakh", "distance_to_acq_branch_km",
                               "branch_tier", "recommended_action", "annual_saving_lakh"]], height=360)

        # ------------------------------------------------------------------ plan
        with tabs[3]:
            a, b = st.columns([1.2, 1])
            w = res.waves
            fig = px.bar(w, x="wave", y="customers", color="wave", text="share_pct", title="Customer migration waves",
                         color_discrete_sequence=[C.PRIMARY, C.BEFORE, C.DURING, C.AFTER], labels={"wave": "", "customers": "Customers (sample)"})
            fig.update_traces(texttemplate="%{text:.0f}%")
            fig.update_layout(showlegend=False)
            with a:
                th.show_chart(th.style_fig(fig, 380), key="int_waves")
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=cx["score"], title=dict(text="Integration complexity"),
                                           gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=C.PRIMARY),
                                                      steps=[dict(range=[0, 35], color="#1f4d33"), dict(range=[35, 60], color="#5a4a14"),
                                                             dict(range=[60, 100], color="#5a1f1c")])))
            with b:
                th.show_chart(th.style_fig(gauge, 380), key="int_gauge")
            th.label("What drives the complexity score")
            th.html_table(pd.DataFrame({
                "Driver": ["High-touch customers (waves 3-4)", "Branches merged / shut", "Products harmonised / sunset", "Core-banking platform gap"],
                "Value": [f"{cx['high_touch_customers'] * 100:.0f}%", f"{cx['branch_changes'] * 100:.0f}%", f"{cx['product_changes'] * 100:.0f}%",
                          f"{cx['target_platform']} → {cx['acquirer_platform']}"],
                "Weight": ["30%", "25%", "20%", "25%"]}))
            st.write("")
            plan = res.customers[["customer_id", "segment", "migration_wave", "migration_approach", "primary_branch_id", "home_city",
                                  "branch_closure_flag", "product_sunset_flag"]]
            th.download_excel(plan, "⬇ Download customer migration plan (Excel)", f"OneRun_Integration_Plan_{bank_id}.xlsx", "dl_int_plan", "Migration_Plan")
            st.caption("The branch-consolidation and product-sunset flags produced here are passed to **Customer Retention Intelligence** as churn-risk drivers.")
