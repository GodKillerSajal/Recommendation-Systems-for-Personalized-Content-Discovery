Write-Host "Running Netflix RecSys analysis pipeline..."

# Activate virtual environment if exists
if (Test-Path "venv\Scripts\activate.ps1") {
    . "venv\Scripts\activate.ps1"
}

# Install requirements (idempotent)
python -m pip install -r requirements.txt

# Step 1: Download data (if not already present)
python src\download_data.py

# Step 2: Preprocess data to Parquet
python src\preprocess.py

# Step 3: Train models
python src\train.py

# Step 4: Run PDF report generation
python src\generate_report.py

# Step 5: Run PDF presentation slide deck generation
python src\generate_presentation.py

Write-Host "Pipeline completed. Deliverables ready in reports/ directory."
