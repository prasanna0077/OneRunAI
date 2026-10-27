"""Platform-level pages: Home, Unified Architecture, Roadmap, Data Lab, Credits."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from onerun import services
from onerun.config import (ACQ_FILE, ACQUIRER, ASSUMPTIONS, CREDITS, DATA_DIR, DATA_FILES, INT_FILE, RET_FILE, SLOGAN,
                           VALUE_PROPOSITION, C, STAGE_COLOURS)
from onerun.data_generation.generator import generate_all
from onerun.modules import registry
from onerun.ui import context as ctx
from onerun.ui import theme as th


# ----------------------------------------------------------------------------- home
def home() -> None:
    st.markdown(th.brand_block(), unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns(3)
    a.markdown(th.card("Who We Are", "<p>OneRun AI is a FinTech intelligence company building the future of intelligent financial transformation.</p>"), unsafe_allow_html=True)
    b.markdown(th.card("What We Do", "<p>Empower banks, NBFCs, and financial institutions with AI-driven decision intelligence.</p>"), unsafe_allow_html=True)
    c.markdown(th.card("Why We Exist", "<p>Transform complex M&A decisions into data-driven value creation.</p>"), unsafe_allow_html=True)
    st.write("")
    st.markdown(f"<div class='or-label'>Value Proposition</div><div style='color:#fff;font-size:17px'>{VALUE_PROPOSITION}</div>", unsafe_allow_html=True)
    st.write("")

    th.label("The MVP - one run across the deal lifecycle")
    steps = "".join(
        f"<div class='step' style='background:{STAGE_COLOURS[m.stage]}'><h4>{m.icon} {m.stage}</h4><b>{m.title}</b><ul><li>{m.description}</li>"
        f"<li>{m.engines}</li></ul></div>" + ("<div class='arrow'>➜</div>" if i < 2 else "")
        for i, m in enumerate(registry.MVP_MODULES))
    st.markdown(f"<div class='or-flow'>{steps}</div>", unsafe_allow_html=True)
    st.write("")

    with st.spinner("Loading deal snapshot ..."):
        tid = ctx.current_target_id()
        k = services.deal_context(tid)["kpis"]
        t = services.target_row(tid)
    th.label(f"Live deal snapshot - {ACQUIRER['name']} × {t.bank_name}")
    th.kpi_row([
        ("Acquisition score", f"{k['acquisition_score']:.0f}/100", k["tier"], C.BEFORE),
        ("Risk-adj. synergy", th.inr_cr(k["risk_adj_synergy_cr"]), "per year", C.GREEN),
        ("Integration complexity", f"{k['complexity_score']:.0f}/100", k["complexity_band"], C.DURING),
        ("Expected churn", f"{k['expected_churn_pct']:.1f}%", "12 months post-merger", C.AFTER),
    ])
    st.write("")
    th.label("Outcomes")
    th.checks(["Better acquisition decisions", "Faster due diligence & deal execution", "Reduced integration & operation risk",
               "Lower customer attrition", "Accelerated synergy realization", "Increased enterprise value"])
    st.caption("All institutions and figures are synthetic and for demonstration only.")


# ----------------------------------------------------------------------------- architecture
def architecture() -> None:
    th.page_header("Unified Platform Architecture", "Across the Lifecycle", "How the layers fit together.", "🏗")
    layers = [
        ("Experience layer", ["Streamlit app", "Our Bank profile", "Deal-context selector", "Dataset & Model Summary", "Excel exports"]),
        ("Intelligence modules (plug-ins)", [f"{m.icon} {m.title}" for m in registry.ALL_MODULES if m.status in ("live", "foundation")]
         + ["＋ 11 roadmap modules (registry.py)"]),
        ("Service layer", ["services.py", "Deal Context", "Cached model training", "MVP-to-MVP hand-off flags", "Pro-forma combined bank"]),
        ("AI / ML engines", ["Random Forest", "Gradient Boosting", "Decision Trees", "Logistic Regression", "K-Means", "Hierarchical (Ward)",
                             "Narrative engine (template → GenAI)"]),
        ("Data layer", [f.name for f in DATA_FILES] + ["Synthetic data generators (onerun/data_generation)"]),
    ]
    for title, boxes in layers:
        st.markdown(f"<div class='or-layer'><div class='t'>{title}</div>{''.join(f'<span class=or-box>{b}</span>' for b in boxes)}</div>",
                    unsafe_allow_html=True)
    th.label("Data flow between the MVPs")
    st.markdown("**Acquisition** picks the target → **Integration** clusters its customers / products / branches and emits "
                "*branch-closure* and *product-sunset* flags → **Retention** uses those flags as churn drivers → "
                "**Executive** (Phase 2) summarises all three via `services.deal_context()`.")


# ----------------------------------------------------------------------------- roadmap
def roadmap() -> None:
    th.page_header("Product Intelligence Portfolio", "Across the Lifecycle", None, "🗺")
    st.caption("AI-Powered Intelligence Across the M&A Lifecycle")
    status_txt = {"live": "✔ MVP", "foundation": "◐ Foundation", "planned": "○ Planned"}
    rows = []
    for i, m in enumerate(registry.ALL_MODULES, 1):
        rows.append({"#": i, "Intelligence": f"{m.icon} {m.title}", "Stage": m.stage, "Key business question": m.question,
                     "AI/ML engine": m.engines, "Phase": "MVP" if m.phase == 1 else f"Phase {m.phase}" if m.phase < 4 else "Backlog",
                     "Status": status_txt.get(m.status, m.status)})
    th.show_df(pd.DataFrame(rows), height=560)
    st.write("")
    c1, c2, c3 = st.columns(3)
    for col, ph, name in ((c1, 1, "MVP"), (c2, 2, "Phase 2"), (c3, 3, "Phase 3")):
        items = "".join(f"<li>{m.title}</li>" for m in registry.by_phase(ph))
        col.markdown(th.card(name, f"<ul>{items}</ul>"), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data lab
def data_lab() -> None:
    th.page_header("Data Lab", "Across the Lifecycle", None, "🧪")
    st.markdown("Regenerate the synthetic Excel datasets with your own size and seed, or download the current files.")
    c1, c2, c3 = st.columns(3)
    seed = c1.number_input("Random seed", 0, 99999, 2026)
    banks = c1.slider("Candidate institutions", 15, 80, 40)
    cpb = c2.slider("Customers per institution", 100, 600, 250, 50)
    deals = c2.slider("Historical deals", 300, 2000, 700, 100)
    hist = c3.slider("Past-merger customers", 3000, 20000, 8000, 1000)
    if st.button("♻ Regenerate datasets", type="primary"):
        with st.spinner("Generating Excel workbooks ..."):
            generate_all(int(seed), banks, cpb, deals, hist)
            services.clear_caches()
            for k in list(st.session_state):
                if k.startswith("deal_"):
                    del st.session_state[k]
        st.success("Datasets regenerated - models will retrain on next page view.")
        st.rerun()
    st.write("")
    for f in DATA_FILES:
        if f.exists():
            with st.expander(f"{f.name}  ·  {f.stat().st_size / 1e6:.1f} MB"):
                sheets = pd.read_excel(f, sheet_name=None)
                st.write({k: f"{len(v):,} rows × {len(v.columns)} cols" for k, v in sheets.items()})
                st.download_button(f"⬇ Download {f.name}", f.read_bytes(), file_name=f.name, key=f"dl_{f.name}")
    th.label("Business assumptions in use (edit in onerun/config.py)")
    th.show_df(pd.DataFrame({"assumption": list(ASSUMPTIONS), "value": list(ASSUMPTIONS.values())}))
    th.label("Acquirer profile (fictional client)")
    th.show_df(pd.DataFrame([(k, str(v)) for k, v in ACQUIRER.items()], columns=["attribute", "value"]))


# ----------------------------------------------------------------------------- credits
def credits() -> None:
    cr = CREDITS
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown(th.brand_block(), unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:26px;font-style:italic;color:#fff;margin-top:30px'>{cr['thanks']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{C.PRIMARY};font-style:italic;margin-top:24px'>{SLOGAN}</div>", unsafe_allow_html=True)
    with right:
        def block(title, items):
            lis = "".join(f"<li>{i}</li>" for i in items)
            st.markdown(f"<div class='or-label'>{title}</div><ul style='color:#fff'>{lis}</ul>", unsafe_allow_html=True)
        st.markdown(f"<div class='or-label' style='font-size:18px'>With Gratitude To</div>", unsafe_allow_html=True)
        block("Faculty Mentors", cr["faculty"])
        block("Academic Partners", cr["partners"])
        block("Peers & Collaborators", cr["peers"])
        extra = "".join(f"<div style='color:#fff'>{x}</div>" for x in (cr["contact"],) if x)
        link = f"<div><a href='{cr['linkedin']}' target='_blank'>LinkedIn</a></div>" if cr["linkedin"] else ""
        st.markdown(f"<div class='or-label'>Presented By</div><div style='color:#fff'>{cr['presented_by']}</div>"
                    f"<div style='color:{C.MUTED}'>{cr['programme_id']} · {cr['date']}</div>{extra}{link}", unsafe_allow_html=True)
