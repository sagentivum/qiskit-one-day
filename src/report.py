from __future__ import annotations
import json
import pandas as pd
from pathlib import Path

def save_results(row: dict, outdir: str = "results"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # json
    with open(Path(outdir) / "results.json", "w") as f:
        json.dump(row, f, indent=2)
    # csv (single-row)
    pd.DataFrame([row]).to_csv(Path(outdir) / "results.csv", index=False)

