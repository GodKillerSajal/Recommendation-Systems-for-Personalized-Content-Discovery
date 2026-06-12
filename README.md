# Netflix RecSys Analysis

A reproducible end-to-end recommendation pipeline built on the **Netflix Prize** dataset. The project parses, preprocesses, models, evaluates, and compiles results in a professional PDF technical report and slide deck.

## Repository Structure
```
netflix-recsys-analysis/
├─ README.md                    # Project overview & guide (this file)
├─ requirements.txt             # Python dependencies
├─ .gitignore                   # Ignore raw files and model binaries
├─ notebook.ipynb               # Full Jupyter Notebook containing EDA, modeling and evaluation
├─ data/
│   ├─ raw/                     # Raw Netflix txt files downloaded via Kaggle CLI
│   └─ processed/               # Parquet files, user/item index maps, metrics.json
├─ reports/
│   ├─ report.pdf               # Final technical report (PDF)
│   └─ presentation.pdf         # Final 8-slide presentation deck (ppt)
├─ scripts/
│   └─ run_all.ps1              # Powershell orchestration script
└─ src/
    ├─ download_data.py         # Configures Kaggle API credentials
    ├─ preprocess.py            # Converts raw txt interactions to optimized Parquet
    ├─ train.py                 # Trains SVD & ALS, computes RMSE & MAP@10
```

## Setup & Execution

### 1. Configure Python Environment
Create a Python virtual environment and install dependencies:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Kaggle Credentials
Place the provided Kaggle credentials token (`key: `) in your local `.kaggle` directory. The pipeline script does this automatically at runtime.

### 3. Run Pipeline
Execute the full workflow (downloading data, preprocessing, modeling, evaluation, and report/presentation creation):
```powershell
.\scripts\run_all.ps1
```

## Modeling Insights & Performance
- **SVD RMSE**: `0.9896` (Optimized for explicit ratings prediction).
- **ALS RMSE**: `2.8183` (Models user preference rather than explicit ratings, leading to scale mismatch).
- **ALS MAP@10**: `0.0057` (Optimized for implicit ranking).

## License
MIT License.
