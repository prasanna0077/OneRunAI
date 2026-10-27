# OneRun AI - M&A Intelligence Platform
*One Run. Every M&A Decision.*

Streamlit + pandas + scikit-learn implementation of the OneRun AI capstone pitch (3 MVPs, Excel datasets,
brand theme, foundation for Phase 2 / 3).

| MVP | Stage | Question | Engine |
|---|---|---|---|
| Acquisition Intelligence | Before the deal | Which institution should we acquire? | Random Forest, Gradient Boosting, Decision Tree |
| Integration Intelligence | During the deal | How do we integrate customers, products, operations? | K-Means, Hierarchical (Ward) |
| Customer Retention Intelligence | After the deal | Which customers will leave? | Logistic Regression, Random Forest, Gradient Boosting |

The three are **chained**: the target picked in Acquisition drives Integration; Integration's
branch-closure / product-sunset flags become churn drivers in Retention.

## Run it in PyCharm (Python 3.10+)
1. **File > Open** the `OneRunAI` folder.
2. **Settings > Project > Python Interpreter > Add Interpreter > Virtualenv** (new `.venv`).
3. Open the PyCharm **Terminal** and run: `pip install -r requirements.txt`
4. *(Optional)* right-click `generate_data.py` > Run - rebuilds the Excel files in `data/`
   (they are also created automatically on first launch; sample files are already included).
5. Start the app: in the Terminal run `streamlit run app.py`
   *or* create a Run Configuration: **Run > Edit Configurations > + > Python**, choose **Module name** = `streamlit`,
   **Parameters** = `run app.py`, **Working directory** = project root.
6. Browser opens at http://localhost:8501.

`python tools/ui_mock_check.py` runs every page's Python logic against a mock Streamlit (catches wrong columns / pandas errors; it cannot judge visuals).
`python smoke_test.py` runs the whole data > ML chain headless (no Streamlit) and prints a health report.

## Pages
Home · **Our Bank** (profile, KPIs, balance-sheet mix, footprint, benchmark) · Acquisition · Integration · Retention ·
**Executive Intelligence** (verdict, threshold legend, post-merger combined bank) · **Dataset & Model Summary** ·
Platform Architecture · Roadmap · Phase 2/3 placeholders · Data Lab · Credits.

Thresholds (acquisition tiers, complexity bands, churn limit) are in `onerun/config.py` (`ASSUMPTIONS`, `THRESHOLDS`).
Meridian's rupee amounts live in the Excel sheet `Acquirer_Financials` (edit it to change Our Bank).
Deployment options (Streamlit Cloud, Docker) are in **DEPLOYMENT.md**.

## Project layout
```
app.py                     Streamlit entry point + navigation
generate_data.py           CLI to (re)generate datasets  (--seed --banks --customers-per-bank ...)
smoke_test.py              headless end-to-end check
data/                      3 Excel workbooks (README + data + Data_Dictionary sheets each)
onerun/config.py           brand colours, acquirer profile, assumptions, CREDITS  <- edit here
onerun/data_generation/    realistic synthetic data (see below)
onerun/ml/                 acquisition.py, integration.py, retention.py, common.py
onerun/services.py         loads Excel, trains/caches models, builds the Deal Context
onerun/modules/            one class per module + registry.py
onerun/ui/                 theme.py, context.py, pages.py, our_bank_page.py, summary_page.py
onerun/ml/proforma.py      combined-bank (Day-1 merge) maths
onerun/ai/narrative.py     Executive narrative engine (template now, LLM later)
```

## How the datasets are made realistic
* A hidden *quality* factor per bank correlates ROA, CASA, GNPA, PCR, CAR, digital and tech metrics.
* Four institution types (private, small-finance, urban co-op, old-gen) with distinct profiles; log-normal sizes.
* Historical deals: success driven by fundamentals plus a "sweet-spot" effect for size and geographic overlap, with noise.
* Customers come from 6 hidden archetypes (never stored) so clustering has to rediscover them; missing bureau scores / survey answers included.
* Past-merger churn is driven by tenure, products, engagement, complaints, branch closure, product sunset, fee change.
* All entities are fictional; same `--seed` gives the same data.

Hold-out results on the shipped data: acquisition AUC ~0.86, churn AUC ~0.79.

## Roadmap hooks (already in place)
* **Add a module:** subclass `BaseModule` in `onerun/modules/`, implement `render()`, register in `registry.py`;
  navigation, roadmap and architecture pages update automatically.
* **Phase 2:** Executive Intelligence is a working foundation; Risk & Due Diligence and Synergy are registered placeholders.
* **Phase 3:** GenAI Deal Assistant, Regulatory Intelligence, Board Reporting are placeholders.
  `services.deal_context(bank_id)` is the single input for them; swap `get_engine()` in `onerun/ai/narrative.py` for an LLM engine.
* Credits page: edit `CREDITS` in `onerun/config.py` (add your LinkedIn / contact there).

## Notes
* Business assumptions (synergy %, branch savings, tier cut-offs, playbook costs) are illustrative, not market data.
* To use your own data, keep the sheet and column names (see each workbook's `Data_Dictionary`).
