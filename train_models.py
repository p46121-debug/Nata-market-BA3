"""
Re-train every model and rebuild the analytics bundle, then save them as .sav files in ./models.

Usage (from the repository folder):
    python train_models.py                      # expects data/nata_supermarket_data.xlsx
    python train_models.py path/to/file.xlsx

Run this again whenever the data changes or you upgrade scikit-learn / pandas.
"""
import sys
import time
from pathlib import Path

import joblib

from nata_core import load_raw, train_all

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/nata_supermarket_data.xlsx")
if not src.exists():
    sys.exit(f"Data file not found: {src}. Put the case Excel file there or pass its path.")

t0 = time.time()
art = train_all(load_raw(src), progress=lambda m, p: print(f"[{p:>4.0%}] {m}"))
out = Path("models"); out.mkdir(exist_ok=True)
for key, fname in [("segmentation", "segmentation_model.sav"), ("response", "response_model.sav"),
                   ("demand", "demand_models.sav"), ("results", "analytics_results.sav")]:
    joblib.dump(art[key], out / fname, compress=3)
    print(f"saved models/{fname} ({(out / fname).stat().st_size / 1024:,.0f} KB)")
print(f"Finished in {time.time() - t0:.0f}s")
