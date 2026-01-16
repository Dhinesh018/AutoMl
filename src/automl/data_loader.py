import pandas as pd
from pathlib import Path

def load_dataset(csv_path: str, target_column: str):
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    return df
