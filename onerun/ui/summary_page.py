"""Dataset & Model Summary - what data each intelligence uses, how the models perform, and what it tells us."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from onerun import services
from onerun.config import ACQ_FILE, C, INT_FILE, RET_FILE
from onerun.ml import acquisition as acq_ml
from onerun.ml import retention as ret_ml
from onerun.ui import context as ctx
from onerun.ui import theme as th

SKIP_SHEETS = ("README", "Data_Dictionary")


@st.cache_data(show_spinner=False)
def _sheets(path: str, version: float) -> dict:
    return {k: v for k, v in pd.read_excel(path, sheet_name=None).items() if k not in SKIP_SHEETS}


def _sheet_table(sheets: dict) -> pd.DataFrame:
    return pd.DataFrame([dict(Sheet=k, Rows=len(v), Columns=v.shape[1], **{"Missing %": float(v.isna().mean().mean() * 100)})
                         for k, v in sheets.items()])


def _insights(items: list[str]) -> None:
    th.label("Insights")
    st.markdown("\n".join(f"- {i}" for i in items))
    st.caption("The data is synthetic, so these findings illustrate what the platform surfaces - they are not market facts.")


def _model_block(suite, key: str, importance_labels: dict, colour: str, threshold_note: str) -> None:
    m = suite.metrics.copy()
    long = m.melt(id_vars="Model", value_vars=["AUC", "Accuracy", "Precision", "Recall", "F1"], var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", title="Model comparison (hold-out set)",
                 color_discrete_sequence=C.SEQ)
    fig.update_yaxes(range=[0, 1])
    c1, c2 = st.columns([1.3, 1])
    with c1:
        th.show_chart(th.style_fig(fig, 380, legend_bottom=True), key=f"sum_cmp_{key}")
    with c2:
        th.show_df(m.round(3))
        st.caption(f"Trained on {suite.n_train:,} rows, tested on {suite.n_test:,}. {threshold_note}")
    a, b = st.columns(2)
    cm = suite.confusion[suite.best_model]
    heat = px.imshow(cm, text_auto=True, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"],
                     color_continuous_scale=[[0, "#2F2F2F"], [1, colour]], title=f"Confusion matrix - {suite.best_model}")
    heat.update_coloraxes(showscale=False)
    with a:
        th.show_chart(th.style_fig(heat, 340), key=f"sum_cm_{key}")
    imp = suite.importance.mean(axis=1).sort_values(ascending=False).head(10)
    imp.index = [importance_labels.get(i, i) for i in imp.index]
    fi = px.bar(imp.iloc[::-1], orientation="h", title="Top features (avg. importance)", labels={"value": "Importance", "index": ""},
                color_discrete_sequence=[colour])
    fi.update_layout(showlegend=False)
    with b:
        th.show_chart(th.style_fig(fi, 340), key=f"sum_imp_{key}")


def render() -> None:
    th.page_header("Dataset & Model Summary", "Across the Lifecycle", None, "📚")
    ver = services.data_version()
    acq_sheets, int_sheets, ret_sheets = _sheets(str(ACQ_FILE), ver), _sheets(str(INT_FILE), ver), _sheets(str(RET_FILE), ver)
    acq_suite, scored = services.acquisition_results()
    ret_suite = services.retention_suite()
    hist = acq_sheets["Historical_Deals"]
    churn_hist = ret_sheets["Historical_Merger_Customers"]
    bank_id = ctx.current_target_id()
    integ = services.integration_results(bank_id)
    tname = services.target_row(bank_id).bank_name

    # ------------------------------------------------------------------ shared insight maths
    a_imp = acq_suite.importance.mean(axis=1).sort_values(ascending=False)
    r_imp = ret_suite.importance.mean(axis=1).sort_values(ascending=False)
    a_top = [acq_ml.FEATURE_LABELS.get(i, i) for i in a_imp.index[:3]]
    r_top = [ret_ml.FEATURE_LABELS.get(i, i) for i in r_imp.index[:3]]
    am, rm = acq_suite.metrics, ret_suite.metrics
    a_lead = am.loc[0, "AUC"] - am.loc[1, "AUC"]
    r_lead = rm.loc[0, "AUC"] - rm.loc[1, "AUC"]
    success_by_type = hist.groupby("bank_type").synergy_realized.mean().sort_values(ascending=False) * 100
    base = lambda col: churn_hist.groupby(col).churned_12m.mean() * 100      # noqa: E731
    br, ps, fe = base("branch_closure_flag"), base("product_sunset_flag"), base("fee_increase_flag")

    tabs = st.tabs(["🧭 Overview", "🎯 Acquisition", "🧬 Integration", "🛡 Retention"])

    # ------------------------------------------------------------------ overview
    with tabs[0]:
        total_rows = lambda d: sum(len(v) for v in d.values())      # noqa: E731
        overview = pd.DataFrame([
            dict(Intelligence="Acquisition", Stage="Before the Deal", Workbook=ACQ_FILE.name, Rows=total_rows(acq_sheets),
                 Engine="Random Forest · Gradient Boosting · Decision Tree", Best=am.loc[0, "Model"], **{"Best AUC": am.loc[0, "AUC"]}),
            dict(Intelligence="Integration", Stage="During the Deal", Workbook=INT_FILE.name, Rows=total_rows(int_sheets),
                 Engine="K-Means · Hierarchical (Ward)", Best="K-Means", **{"Best AUC": float("nan")}),
            dict(Intelligence="Retention", Stage="After the Deal", Workbook=RET_FILE.name, Rows=total_rows(ret_sheets),
                 Engine="Logistic Regression · Random Forest · Gradient Boosting", Best=rm.loc[0, "Model"], **{"Best AUC": rm.loc[0, "AUC"]}),
        ])
        th.kpi_row([
            ("Workbooks", "3", "Excel, with README + data dictionary", C.PRIMARY),
            ("Total rows", f"{overview.Rows.sum():,}", "across all sheets", C.GREEN),
            ("Models trained", f"{len(acq_suite.models) + len(ret_suite.models) + 3}", "6 supervised + 3 clustering", C.BEFORE),
            ("Best AUC", f"{max(am.loc[0, 'AUC'], rm.loc[0, 'AUC']):.2f}", "hold-out performance", C.AFTER),
        ])
        st.write("")
        th.show_df(overview.round(3))
        _insights([
            f"**Acquisition** - {am.loc[0, 'Model']} leads with AUC {am.loc[0, 'AUC']:.2f}, {a_lead * 100:.1f} points ahead of the runner-up; "
            f"the biggest success drivers are {', '.join(a_top)}.",
            f"**Retention** - {rm.loc[0, 'Model']} leads with AUC {rm.loc[0, 'AUC']:.2f}; churn is mostly explained by {', '.join(r_top)}.",
            f"**Integration** - {len(integ.segments)} customer segments found for {tname}; "
            f"{int(integ.products.product_sunset_flag.sum())} of {len(integ.products)} products flagged to sunset.",
            "Churn is hard to predict exactly because customer behaviour is noisy, so an AUC near 0.8 is a realistic level.",
        ])

    # ------------------------------------------------------------------ acquisition
    with tabs[1]:
        th.label("Dataset")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            th.show_df(_sheet_table(acq_sheets).round(1))
        with c2:
            sb = px.bar(success_by_type.reset_index(), x="synergy_realized", y="bank_type", orientation="h",
                        title="Share of historical deals that realised synergy (%)", labels={"synergy_realized": "%", "bank_type": ""},
                        color_discrete_sequence=[C.BEFORE])
            th.show_chart(th.style_fig(sb, 280), key="sum_acq_type")
        c1, c2 = st.columns(2)
        h1 = px.histogram(hist, x="roa_pct", color=hist.synergy_realized.map({1: "Synergy realised", 0: "Not realised"}), nbins=30,
                          barmode="overlay", opacity=.7, title="ROA of target vs. outcome", labels={"roa_pct": "ROA %", "color": ""},
                          color_discrete_sequence=[C.GREEN, C.RED])
        h2 = px.histogram(hist, x="gnpa_pct", color=hist.synergy_realized.map({1: "Synergy realised", 0: "Not realised"}), nbins=30,
                          barmode="overlay", opacity=.7, title="Gross NPA of target vs. outcome", labels={"gnpa_pct": "GNPA %", "color": ""},
                          color_discrete_sequence=[C.GREEN, C.RED])
        with c1:
            th.show_chart(th.style_fig(h1, 300, legend_bottom=True), key="sum_acq_h1")
        with c2:
            th.show_chart(th.style_fig(h2, 300, legend_bottom=True), key="sum_acq_h2")
        th.label("Models")
        _model_block(acq_suite, "acq", acq_ml.FEATURE_LABELS, C.BEFORE, "Classification threshold 0.5.")
        tier_mix = scored.tier.value_counts()
        _insights([
            f"{am.loc[0, 'Model']} is best (AUC {am.loc[0, 'AUC']:.2f}); the simple Decision Tree is "
            f"{(am.loc[0, 'AUC'] - am.set_index('Model').loc['Decision Tree', 'AUC']) * 100:.1f} AUC points behind but easiest to explain.",
            f"Top drivers of a successful deal: {', '.join(a_top)}.",
            f"{success_by_type.index[0]} deals realised synergy most often ({success_by_type.iloc[0]:.0f}%); "
            f"{success_by_type.index[-1]} least ({success_by_type.iloc[-1]:.0f}%).",
            f"Of {len(scored)} live candidates: " + ", ".join(f"{v} {k}" for k, v in tier_mix.items()) + ".",
        ])

    # ------------------------------------------------------------------ integration
    with tabs[2]:
        th.label(f"Dataset (current deal target: {tname})")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            th.show_df(_sheet_table(int_sheets).round(1))
        with c2:
            st.markdown("**Unsupervised models** - there is no label to predict, so quality is judged by cluster separation (silhouette) "
                        "and by whether the groups make business sense.")
        c1, c2, c3 = st.columns(3)
        sil = px.line(integ.silhouette, x="k", y="silhouette", markers=True, title="Silhouette by number of segments",
                      labels={"k": "k", "silhouette": "Silhouette"}, color_discrete_sequence=[C.DURING])
        pc = integ.products.cluster.value_counts().reset_index()
        pc.columns = ["Cluster", "Products"]
        pb = px.bar(pc, x="Products", y="Cluster", orientation="h", title="Products per cluster", color_discrete_sequence=[C.PRIMARY])
        bt = integ.branches.branch_tier.value_counts().reset_index()
        bt.columns = ["Tier", "Branches"]
        bb = px.bar(bt, x="Tier", y="Branches", title="Branches per tier", color_discrete_sequence=[C.GREEN])
        with c1:
            th.show_chart(th.style_fig(sil, 320), key="sum_int_sil")
        with c2:
            th.show_chart(th.style_fig(pb, 320), key="sum_int_prod")
        with c3:
            th.show_chart(th.style_fig(bb, 320), key="sum_int_br")
        th.show_df(integ.segments[["segment", "customers", "share_pct", "avg_age", "avg_balance_inr", "avg_logins_30d", "wave"]].round(1))
        best_k = int(integ.silhouette.loc[integ.silhouette.silhouette.idxmax(), "k"])
        big = integ.segments.sort_values("customers", ascending=False).iloc[0]
        high_touch = float((integ.customers.migration_wave >= 3).mean() * 100)
        _insights([
            f"Silhouette is highest at k = {best_k} ({integ.silhouette.silhouette.max():.2f}); the app default of 5 is a business-readable choice.",
            f"Largest segment: **{big.segment}** ({big.share_pct:.0f}% of sampled customers).",
            f"{high_touch:.0f}% of customers need relationship-led or assisted migration (waves 3-4), which drives integration effort.",
            f"{int(integ.branches.branch_closure_flag.sum())} of {len(integ.branches)} sampled branches are merged or shut; "
            f"{int(integ.products.product_sunset_flag.sum())} products are flagged to sunset.",
        ])

    # ------------------------------------------------------------------ retention
    with tabs[3]:
        th.label("Dataset")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            th.show_df(_sheet_table(ret_sheets).round(1))
        with c2:
            rf = pd.DataFrame({"Integration event": ["Branch consolidated", "Product sunset", "Fee increase"],
                               "Churn if yes %": [br.get(1, 0), ps.get(1, 0), fe.get(1, 0)],
                               "Churn if no %": [br.get(0, 0), ps.get(0, 0), fe.get(0, 0)]})
            rl = rf.melt(id_vars="Integration event", var_name="Case", value_name="Churn %")
            rb = px.bar(rl, x="Integration event", y="Churn %", color="Case", barmode="group", title="Churn rate by integration event (history)",
                        color_discrete_sequence=[C.AFTER, C.MUTED])
            th.show_chart(th.style_fig(rb, 300, legend_bottom=True), key="sum_ret_events")
        th.label("Models")
        _model_block(ret_suite, "ret", ret_ml.FEATURE_LABELS, C.AFTER, "Classification threshold 0.35 (favours catching churners).")
        _insights([
            f"Overall 12-month churn in past mergers: {ret_suite.base_rate * 100:.0f}%.",
            f"Customers whose branch was consolidated churn at **{br.get(1, 0):.0f}%** vs {br.get(0, 0):.0f}% otherwise "
            f"({br.get(1, 0) / max(br.get(0, 1), 1):.1f}x); product sunset: {ps.get(1, 0):.0f}% vs {ps.get(0, 0):.0f}%; "
            f"fee increase: {fe.get(1, 0):.0f}% vs {fe.get(0, 0):.0f}%.",
            f"Top churn drivers: {', '.join(r_top)}.",
            f"{rm.loc[0, 'Model']} is best (AUC {rm.loc[0, 'AUC']:.2f}), {r_lead * 100:.1f} points ahead of the runner-up"
            + (" - close enough that the simplest, most explainable model is a fair choice." if r_lead < .03 else "."),
        ])
