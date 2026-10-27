"""Developer check: runs every page's real Python logic against a *mock* Streamlit/Plotly.

Catches wrong column names, pandas errors, bad KPI tuples and similar bugs without needing a browser.
It cannot check visual rendering - always open the real app once.   Run:  python tools/ui_mock_check.py
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ------------------------------------------------------------------ fake streamlit
st = MagicMock(name="streamlit")
st.__version__ = "1.56.0"
st.session_state = {}
def _widget():
    """A container (column / tab) whose widget methods behave like the top-level st.* ones."""
    w = MagicMock()
    for name in ("selectbox", "multiselect", "slider", "button", "number_input", "radio", "checkbox"):
        setattr(w, name, getattr(st, name))
    return w


st.columns.side_effect = lambda spec, **k: [_widget() for _ in range(spec if isinstance(spec, int) else len(spec))]
st.tabs.side_effect = lambda labels, **k: [_widget() for _ in labels]
st.number_input.side_effect = lambda label, *a, **k: a[2] if len(a) > 2 else 0
st.selectbox.side_effect = lambda label, options, index=0, **k: list(options)[index]
st.multiselect.side_effect = lambda label, options, default=None, **k: default if default is not None else []
st.slider.side_effect = lambda label, *a, **k: k.get("value", a[2] if len(a) > 2 else a[0])
st.button.return_value = False
st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
sys.modules["streamlit"] = st

# ------------------------------------------------------------------ fake plotly (validates dataframe columns)
ERRORS = []


def _checked(name):
    def fn(*args, **kw):
        df = args[0] if args else kw.get("data_frame")
        if isinstance(df, pd.DataFrame):
            for arg in ("x", "y", "color", "names", "values", "text", "size", "hover_name", "lat", "lon"):
                v = kw.get(arg)
                if isinstance(v, str) and v not in df.columns:
                    ERRORS.append(f"px.{name}: column '{v}' not in {list(df.columns)}")
        return MagicMock(name=f"fig_{name}")
    return fn


px = types.ModuleType("plotly.express")
for n in ("bar", "pie", "line", "scatter", "histogram", "box", "scatter_geo", "imshow"):
    setattr(px, n, _checked(n))
go = MagicMock(name="go")
pl = types.ModuleType("plotly")
pl.express, pl.graph_objects = px, go
ff = MagicMock()
for name, mod in (("plotly", pl), ("plotly.express", px), ("plotly.graph_objects", go), ("plotly.figure_factory", ff)):
    sys.modules[name] = mod
pl.figure_factory = ff

# ------------------------------------------------------------------ run pages
from onerun import services                      # noqa: E402
from onerun.modules import registry             # noqa: E402
from onerun.ui import context as ctx            # noqa: E402
from onerun.ui import our_bank_page, pages, summary_page   # noqa: E402

ctx.target_selector = lambda key, label="": services.top_target_id()
for m in registry.ALL_MODULES:                      # modules imported the selector by module reference
    pass

PAGES = {"home": pages.home, "our_bank": our_bank_page.render, "summary": summary_page.render,
         "architecture": pages.architecture, "roadmap": pages.roadmap, "credits": pages.credits}
for m in registry.ALL_MODULES:
    PAGES[m.key] = m.render

failed = 0
for name, fn in PAGES.items():
    before = len(ERRORS)
    try:
        fn()
        status = "ok" if len(ERRORS) == before else "CHART-COLUMN ERRORS"
    except Exception as e:                              # noqa: BLE001
        status, failed = f"EXCEPTION {type(e).__name__}: {e}", failed + 1
    print(f"{name:<14} {status}")
for e in ERRORS:
    print("  !", e)
print("\nFAILED" if failed or ERRORS else "\nALL PAGES PASSED the mock check")
sys.exit(1 if failed or ERRORS else 0)
