# Deploying OneRun AI

The app is a standard Streamlit project: entry point `app.py`, dependencies in `requirements.txt`, no secrets, no database.
Python 3.10 - 3.12 recommended.

## 1. Run locally (PyCharm or terminal)
```
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py             # opens http://localhost:8501
```
PyCharm: open the folder, add a Virtualenv interpreter, run the three commands in the Terminal tab.
Optional Run Configuration: *Python*, **Module name** `streamlit`, **Parameters** `run app.py`, **Working directory** = project root.

## 2. Free online link - Streamlit Community Cloud (best for demos)
1. Create a GitHub account and a **new repository** (public is simplest; private also works).
2. Upload the project (drag-and-drop in GitHub, or `git push`). Keep `app.py` and `requirements.txt` in the repository root.
   Include `data/` for a fixed dataset, or leave it out - the app regenerates it on first launch (about 20 seconds).
3. Go to https://share.streamlit.io and sign in with GitHub.
4. **Create app** -> choose the repository and branch -> **Main file path:** `app.py`.
5. **Advanced settings** -> Python version 3.11 (or 3.12).
6. **Deploy.** The first build takes a few minutes and gives you a shareable URL.
7. To update: push changes to GitHub - the app redeploys automatically.

## 3. Docker (company server or any cloud VM)
```
docker build -t onerun-ai .
docker run -p 8501:8501 onerun-ai          # then open http://<host>:8501
```
Add `-v $(pwd)/data:/app/data` to keep generated or edited Excel files outside the container.
Put the container behind your organisation's reverse proxy / SSO if the app must not be public.

## 4. Changing the data after deployment
* The three Excel workbooks in `data/` are the data source. Edit them (for example the `Acquirer_Financials` sheet for Our Bank) and restart the app.
* **Data Lab** page: regenerate with a new seed / sizes. On Community Cloud this resets when the app restarts - commit `data/` to make a version permanent.

## 5. Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: onerun` | Run from the project root (the folder that contains `app.py`). |
| `StreamlitInvalidHeightError` | Use the latest `onerun/ui/theme.py` (this release); it never passes `height=None`. |
| Old numbers still show | Restart the app (`Ctrl+C`, run again) so cached models reload. |
| First page load is slow | Models train once (about 10 seconds) and are then cached. |
| Cloud app "went to sleep" | Free apps sleep after inactivity; open the link and wait about 30 seconds. |
| Excel file locked (Windows) | Close the workbook in Excel before regenerating data. |

## 6. Before you make a repository public
All institutions, customers and figures are **synthetic**. Check your programme's rules before publishing the code or the credits page.
