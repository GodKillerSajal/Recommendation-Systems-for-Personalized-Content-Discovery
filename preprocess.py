import os
import re
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def parse_netflix_txt(file_path: Path):
    """Parse Netflix combined_data_X.txt format into user, movie, rating list."""
    users = []
    movies = []
    ratings = []
    dates = []
    
    current_movie = None
    print(f"Parsing {file_path.name}...")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, total=24058263 if "combined_data_1" in file_path.name else None):
            line = line.strip()
            if not line:
                continue
            if line.endswith(':'):
                current_movie = int(line[:-1])
            else:
                parts = line.split(',')
                if len(parts) == 3:
                    users.append(int(parts[0]))
                    movies.append(current_movie)
                    ratings.append(float(parts[1]))
                    dates.append(parts[2])
                    
    return pd.DataFrame({
        'user_id': users,
        'movie_id': movies,
        'rating': ratings,
        'date': dates
    })

def main():
    # Parse only combined_data_1.txt for performance and 16GB RAM limit
    txt_file = RAW_DIR / "combined_data_1.txt"
    if not txt_file.exists():
        print(f"File {txt_file} does not exist. Please check dataset download.")
        return
        
    df = parse_netflix_txt(txt_file)
    
    # Cast types to optimize storage & RAM
    df['user_id'] = df['user_id'].astype('int32')
    df['movie_id'] = df['movie_id'].astype('int32')
    df['rating'] = df['rating'].astype('float32')
    df['date'] = pd.to_datetime(df['date'])
    
    out_file = PROCESSED_DIR / "ratings.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_file, compression='snappy')
    print(f"Saved preprocessed ratings to {out_file} ({df.shape[0]} rows)")

if __name__ == "__main__":
    main()
