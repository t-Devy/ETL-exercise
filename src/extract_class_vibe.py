"""
Extract bare bones procedure
- CSVExtractor.extract() returns a pandas DataFrame
- main() demonstrates usage and saves an artifact

Run: python extract_class_vibe.py
"""


from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
import pandas as pd

from src.paths import RAW_CSV_PATH, RAW_PKL_PATH

@dataclass(frozen=True)
class CSVExtractor:
    csv_path: str
    def extract(self) -> pd.DataFrame:

        # Load raw tabular data.
        return pd.read_csv(self.csv_path)



def main() -> None:
    extractor = CSVExtractor(csv_path=str(RAW_CSV_PATH))
    df = extractor.extract()

    df.to_pickle(RAW_PKL_PATH)

    print(f"Extracted rows={len(df)}, cols={len(df.columns)}")
    print(f"Saved raw artifact -> {RAW_PKL_PATH}")
    print(df.head(3))


if __name__ == "__main__":
    main()
