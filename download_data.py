import os
import json
from pathlib import Path

def ensure_kaggle_json():
    """Create kaggle.json with the provided token if it does not exist."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"
    if kaggle_json_path.exists():
        return
    token = {
        "username": "kaggle_user",
        "key": "KGAT_53ae34eec583ba646f64afb63c320a50"
    }
    kaggle_json_path.write_text(json.dumps(token, indent=2))
    # Restrict permissions (Unix style; on Windows this is a no‑op but harmless)
    try:
        os.chmod(kaggle_json_path, 0o600)
    except Exception:
        pass

if __name__ == "__main__":
    ensure_kaggle_json()
    print("Kaggle API token ready at ~/.kaggle/kaggle.json")
