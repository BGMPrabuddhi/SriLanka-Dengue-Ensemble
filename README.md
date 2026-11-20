# Sri Lanka Dengue Early-Warning Dashboard (Ensemble Learning)

> Work in progress: the full README (story, results, diagrams, links) is written in Phase 7.

## Local setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt     # notebooks + training tools
# (requirements.txt alone is enough to run the Streamlit app)
```

**Python version:** models were trained with **Python 3.12** and the exact library pins in
`requirements.txt`. When deploying on Streamlit Community Cloud, open **Advanced settings** and
select **Python 3.12** so the saved `.joblib` models load with the same versions.

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt`
