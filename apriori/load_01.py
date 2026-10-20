"""
load_01.py  -  ONLY loads data (Online Retail transactions). No cleaning, no models, no plots.

Order of attempts:
  1. data/online_retail.csv      (cache made by a previous run)
  2. data/Online Retail.xlsx     (put the UCI file here manually if you like)
  3. download from the UCI repository (cached to CSV afterwards)
  4. FALLBACK: synthetic retail transactions with the same columns
     (only used if the real data cannot be loaded, e.g. no internet)

Real dataset page: https://archive.ics.uci.edu/dataset/352/online+retail
Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
         CustomerID, Country
"""
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_CACHE = DATA_DIR / "online_retail.csv"
XLSX_FILE = DATA_DIR / "Online Retail.xlsx"
SYNTH_FILE = DATA_DIR / "synthetic_retail.csv"
URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def _download():
    print("Downloading Online Retail from UCI (about 23 MB) ...")
    raw = urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".xlsx")][0]
        with zf.open(name) as f:
            return pd.read_excel(f)


def make_synthetic(n_invoices=20000, n_items=250, seed=42):
    """Synthetic baskets: Zipf-like popularity + planted item bundles."""
    rng = np.random.default_rng(seed)
    items = [f"PRODUCT {i:03d}" for i in range(n_items)]
    pop = 1.0 / np.arange(1, n_items + 1) ** 0.9
    pop /= pop.sum()
    bundles = [rng.choice(n_items, size=rng.integers(2, 5), replace=False) for _ in range(25)]
    bundle_prob = rng.uniform(0.02, 0.10, size=len(bundles))

    rows = []
    start = pd.Timestamp("2011-01-01")
    for inv in range(n_invoices):
        basket = set(rng.choice(n_items, size=rng.integers(2, 10), p=pop, replace=False))
        for b, p in zip(bundles, bundle_prob):
            if rng.random() < p:
                basket.update(b.tolist())
        date = start + pd.Timedelta(minutes=int(inv * 25))
        for it in basket:
            rows.append((str(536365 + inv), f"S{it:04d}", items[it], int(rng.integers(1, 6)),
                         date, 2.5, 10000 + inv % 4000, "United Kingdom"))
    return pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                       "InvoiceDate", "UnitPrice", "CustomerID", "Country"])


def load_online_retail():
    """Return (DataFrame, source) where source is 'real' or 'synthetic'."""
    DATA_DIR.mkdir(exist_ok=True)
    if CSV_CACHE.exists():
        return pd.read_csv(CSV_CACHE, dtype={"InvoiceNo": str, "StockCode": str}), "real"
    try:
        df = pd.read_excel(XLSX_FILE) if XLSX_FILE.exists() else _download()
        df.to_csv(CSV_CACHE, index=False)
        return df, "real"
    except Exception as e:  # no internet, blocked, etc.
        print(f"[load_01] Could not load the real dataset ({type(e).__name__}: {e}).")
        print("[load_01] Using SYNTHETIC retail data instead.")
        print(f"[load_01] To use the real data, place 'Online Retail.xlsx' in {DATA_DIR}")
        if SYNTH_FILE.exists():
            return pd.read_csv(SYNTH_FILE, dtype={"InvoiceNo": str, "StockCode": str}), "synthetic"
        df = make_synthetic()
        df.to_csv(SYNTH_FILE, index=False)
        return df, "synthetic"


if __name__ == "__main__":
    df, source = load_online_retail()
    print(f"source={source}  shape={df.shape}")
    print(df.head())
