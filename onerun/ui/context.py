"""Deal Context - the one piece of shared state that links the three MVPs.

Acquisition picks a target -> Integration plans its integration -> Retention scores its customers.
"""
from __future__ import annotations

import streamlit as st

from onerun import services

KEY = "deal_target_id"


def current_target_id() -> str:
    scored = services.acquisition_results()[1]
    tid = st.session_state.get(KEY)
    if tid not in set(scored.bank_id):
        tid = services.top_target_id()
        st.session_state[KEY] = tid
    return tid


def set_target(bank_id: str) -> None:
    st.session_state[KEY] = bank_id


def target_labels() -> dict:
    scored = services.acquisition_results()[1]
    return {r.bank_id: f"#{int(r['rank'])} · {r.bank_name} (score {r.acquisition_score:.0f})" for _, r in scored.iterrows()}


def target_selector(key: str, label: str = "Deal target (from Acquisition Intelligence)") -> str:
    """Selectbox bound to the shared deal context."""
    labels = target_labels()
    ids = list(labels)
    cur = current_target_id()
    choice = st.selectbox(label, ids, index=ids.index(cur), format_func=labels.get, key=f"target_sel_{key}")
    if choice != cur:
        set_target(choice)
    return choice


def short_name(name: str) -> str:
    """'Thar Small Finance Bank' -> 'Thar' (for compact KPI cards)."""
    for suf in (" Small Finance Bank", " Urban Co-operative Bank", " Bank Ltd", " Bank"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name
