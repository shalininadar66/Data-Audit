# Data Audit

Data Audit is a web-based dataset inspection tool designed to help users examine CSV files before using them for machine learning.

## Features

- Upload a CSV file for analysis.
- Inspect dataset quality, including missing values and duplicate records.
- Screen for potential data leakage indicators.
- Review column-level audit results.
- Get recommendations about suitable machine-learning tasks and models based on the available inputs.

## Technology Stack

**Frontend:** React, TypeScript, Vite  
**Backend:** Python, FastAPI, Pandas, NumPy

## Project Structure

- `frontend/` — user interface and frontend configuration.
- `backend/app/` — API and dataset-audit logic.
- `backend/requirements.txt` — Python dependencies.

## Run Locally

### 1. Start the backend

Open a terminal in the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

If PowerShell blocks activation, you can run the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

### 2. Start the frontend

Open a **second terminal** in VS Code:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in the terminal.

## Important Note

Data Audit provides preliminary, heuristic-based checks and recommendations. A flagged issue is not necessarily proof of data leakage, and a suitability score is not a guarantee of machine-learning performance.