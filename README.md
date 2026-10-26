# Data Audit – Dataset Quality & Leakage Analysis Tool

**A web-based tool to evaluate dataset quality, identify potential data leakage risks, and assess Machine Learning task and model suitability — without writing code.**

<p align="center">
  <a href="https://data-audit-peach.vercel.app/"><strong>Live Demo</strong></a> ·
  <a href="https://github.com/shalininadar66/Data-Audit"><strong>GitHub Repository</strong></a>
</p>

---

## Overview

Data Audit is a web application designed to help users understand the quality, structure, and Machine Learning readiness of CSV datasets. Users can upload a dataset through a simple interface and receive an automated audit report containing dataset statistics, data quality indicators, potential leakage signals, and recommendations for suitable Machine Learning tasks and models.

The tool aims to make preliminary dataset auditing more accessible to students, data analysts, data scientists, and Machine Learning practitioners without requiring them to write code or manually inspect every column.

## Live Application

**Try Data Audit:** https://data-audit-peach.vercel.app/

The application is deployed using:
- **Frontend:** Vercel
- **Backend:** Render

> Note: The backend uses Render's free instance, which may spin down after inactivity. The first request after a period of inactivity can take longer to complete.

## Key Features

### 1. Dataset Quality Analysis
- Calculates an overall dataset quality score out of 100.
- Identifies missing values and missing-value percentages.
- Detects duplicate rows.
- Identifies constant columns, empty rows, and ID-like columns.
- Flags high-cardinality categorical columns.
- Examines numeric outliers using the Interquartile Range (IQR) method.
- Checks numeric-like fields for invalid values.
- Identifies inconsistent leading or trailing whitespace in text fields.
- Applies configured range checks to recognized columns, such as age, marks, and percentages.

### 2. Column-Level Statistics
Provides information about individual dataset columns, including:
- Column names and data types.
- Missing-value counts and percentages.
- Unique-value counts.
- Outlier statistics for applicable numeric columns.
- Other structural and data quality indicators.

### 3. Potential Data Leakage Screening
- Detects ID-like columns that may require review before model training.
- Identifies duplicate rows.
- When a target column is supplied, checks for potential direct target copies.
- Flags very strong numeric associations with the target.
- Identifies suspicious target-related column names.

The leakage score is a rule-based risk index based on detected signals. It is **not a probability that leakage exists**. A high correlation or suspicious column is a reason to investigate, not proof of leakage.

### 4. Machine Learning Task Suitability
Assesses how well the observed dataset structure aligns with supported Machine Learning tasks, including:
- Classification
- Regression
- Clustering
- Time-series forecasting
- Anomaly detection
- Dimensionality reduction
- Recommendation systems
- Ranking

The assessment considers relevant dataset characteristics, available target information, feature types, dataset size, and other task-specific indicators.

### 5. Machine Learning Model Recommendations
- Suggests candidate models based on dataset characteristics and task suitability.
- Evaluates model-data alignment using implemented rule-based checks.
- Highlights potential compatibility concerns.
- Provides recommendations related to preprocessing, feature types, target compatibility, and dataset readiness.
- Can compare the selected task with alternative Machine Learning tasks when enabled.

### 6. Simple Web Interface
- Upload and analyse CSV datasets through a browser.
- Select optional analysis inputs, such as a target column and intended Machine Learning task.
- View audit results through a structured dashboard.
- Review scores, statistics, findings, and recommendations without manually writing analysis code.

## How It Works

1. **Upload:** The user uploads a CSV dataset through the web interface.
2. **Read:** The backend reads the CSV into a Pandas DataFrame.
3. **Inspect:** The application analyses the dataset's structure, data types, missing values, duplicates, and other quality indicators.
4. **Assess leakage:** The application checks for rule-based leakage risk signals. Target-dependent checks are performed when a valid target column is provided.
5. **Evaluate suitability:** If task or model information is supplied, the application evaluates the dataset against the implemented suitability criteria.
6. **Generate results:** The application returns the calculated scores, findings, statistics, and recommendations to the frontend.
7. **Display:** The frontend presents the audit results in a readable dashboard.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| Data Processing | Pandas, NumPy |
| API Communication | HTTP, JSON, multipart file uploads |
| Frontend Deployment | Vercel |
| Backend Deployment | Render |
| Version Control | Git, GitHub |

## System Architecture

```text
                 USER
                   |
                   v
          React + TypeScript UI
             (Vercel Hosting)
                   |
                   | HTTP API Request
                   | CSV File Upload
                   v
             FastAPI Backend
             (Render Hosting)
                   |
                   v
         Pandas Data Processing
                   |
          +--------+---------+
          |        |         |
          v        v         v
        Data    Leakage    Task & Model
       Quality   Checks    Suitability
          |        |         |
          +--------+---------+
                   |
                   v
           Audit Report JSON
                   |
                   v
            Frontend Dashboard
```

## Installation and Local Setup

Follow these instructions if you want to run the project on your computer.

### Prerequisites

Install the following:
- Python 3.10 or a compatible version supported by your dependencies.
- Node.js and npm.
- Git.

### 1. Clone the Repository

```bash
git clone https://github.com/shalininadar66/Data-Audit.git
cd Data-Audit
```

### 2. Set Up the Backend

Open a terminal in the project directory:

```bash
cd backend
```

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The backend should now be available at:

- API root: http://127.0.0.1:8000/
- Interactive API documentation: http://127.0.0.1:8000/docs

Keep this terminal running while using the locally hosted frontend.

### 3. Set Up the Frontend

Open a **second terminal** in the project directory.

```bash
cd frontend
npm install
```

Create a `.env` file inside the `frontend` directory with the following variable:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Start the frontend:

```bash
npm run dev
```

Open the local URL displayed in the terminal, usually:

http://localhost:5173/

### 4. Stop the Application

To stop either development server, select its terminal and press:

```text
Ctrl + C
```

## Deployment

The deployed application uses two services:

**Frontend — Vercel**

https://data-audit-peach.vercel.app/

**Backend — Render**

https://data-audit-co27.onrender.com/

The frontend communicates with the backend through the configured `VITE_API_URL` environment variable. For a production deployment, this variable must point to the deployed backend URL, and the backend must allow requests from the deployed frontend origin.

## Scoring Methodology

### Dataset Quality Score

The quality score uses a deterministic, weighted, rule-based approach with a maximum of 100 points.

| Parameter | Weight |
|---|---:|
| Completeness | 22 |
| Duplicate integrity | 18 |
| Outlier burden | 8 |
| Constant columns | 4 |
| High-cardinality columns | 4 |
| ID-like columns | 4 |
| Empty rows | 4 |
| Data type validity | 15 |
| Text consistency | 11 |
| Range validity | 10 |
| **Total** | **100** |

Each category receives a deduction based on the measured data issues, bounded by its assigned weight.

The implemented score bands are:
- **90–100:** Excellent
- **75–89:** Good, with issues to review
- **50–74:** Needs cleaning
- **0–49:** Poor

These scores describe the issues detected by the implemented checks. They do not certify that a dataset is error-free.

### Leakage Risk Score

The leakage score is a rule-based index ranging from 0 to 100:
- Lower scores indicate fewer detected leakage signals.
- Higher scores indicate more or stronger detected signals.
- ID-like columns and duplicate rows contribute to the target-independent component.
- When a valid target is supplied, additional target-dependent checks can contribute to the score.

A score of zero does not guarantee that a dataset is leakage-free. The tool does not replace a proper train-test split, time-aware validation where appropriate, or a review of the data-generation process.

### Task and Model Suitability

Task and model suitability scores are calculated using implemented structural and compatibility rules. They indicate how well the available dataset characteristics match the checks defined by the application.

They are **not trained-model accuracy scores** and do not guarantee that a recommended model will perform well on unseen data.

## Limitations

- The application currently accepts CSV datasets.
- Quality checks are rule-based and may not detect every data issue.
- Outliers may be legitimate observations and require contextual review.
- Leakage screening is based on detectable signals and cannot identify every form of leakage.
- Task and model suitability depend on the implemented rules and the information supplied by the user.
- Range checks apply only to recognized column names with configured rules.
- Model recommendations are preliminary suggestions; actual performance requires preprocessing, model training, suitable validation, and evaluation.
- Large CSV files may require more processing time or exceed the hosting service's resource limits.
- The deployed backend may take longer to respond after inactivity because of free-tier hosting behavior.

## Future Enhancements

Potential future improvements include:
- Support for additional data formats, such as Excel and Parquet.
- More configurable domain-specific validation rules.
- Advanced leakage checks based on train-test overlap and temporal validation.
- Downloadable audit reports in PDF and CSV formats.
- Additional explainability for scores and recommendations.
- More comprehensive model evaluation workflows.
- Enhanced handling of large datasets.
- Automated data-cleaning suggestions and optional preprocessing.

These are possible extensions and are not necessarily implemented in the current version.

## Project Structure

The main application is organized into separate frontend and backend components.

```text
Data-Audit/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── quality_engine.py
│   │   ├── leakage_engine.py
│   │   └── ...
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   └── ...
│   ├── package.json
│   └── ...
└── README.md
```

*Note: The structure above illustrates the main components. Individual files may differ depending on the current repository version.*

## Who Can Use Data Audit?

- Students learning data analysis and Machine Learning.
- Data analysts performing initial dataset inspections.
- Data scientists preparing datasets for modelling.
- Machine Learning practitioners reviewing preliminary data quality and leakage risks.
- Researchers seeking a quick, code-free overview of CSV datasets.

## Contributing

Suggestions, bug reports, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Submit a pull request describing the changes.

## Author

**Shalini Nadar**

BSc Data Science Student

GitHub: https://github.com/shalininadar66

## License

No license has been specified here. If you intend to let others reuse, modify, or distribute this project, consider adding an appropriate open-source license file to the repository.

---

**Data Audit — Understand your data before you build your models.**
