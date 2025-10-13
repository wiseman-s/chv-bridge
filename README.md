# CHV Bridge — Frontend-First MVP (Streamlit)

This is a frontend-focused MVP demo for CHV Bridge. It includes:
- Visit logging (saved to a sample CSV)
- Incentive analytics (configurable rules)
- Time-based filters
- Leaderboard
- CSV and simple PDF report export
- Demo API key generator (frontend-only, 5-minute expiry)

## How to run locally

1. Create a python environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. The app uses `sample_data/visits_sample.csv` as the dataset. When you "Log Visit" it appends to that CSV.

## Project structure
- `app.py` — main Streamlit app (single-file multipage flow)
- `utils/` — helper modules (data ops, incentives, charts, reports)
- `sample_data/visits_sample.csv` — prefilled demo data
- `.streamlit/config.toml` — theme customization

## Next steps
- Replace CSV persistence with a backend (FastAPI + Postgres or Supabase)
- Move API key generation to a backend and validate server-side
- Add authentication and role-based permissions
