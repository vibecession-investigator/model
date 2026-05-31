"""
Fetch economic data from FRED (Federal Reserve Economic Data) and produce
a quarterly CSV dataset from 1980 Q1 through 2025 Q4.

Requires a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set it via environment variable:  export FRED_API_KEY=your_key_here

Sources (all FRED / St. Louis Fed):
  CPI             : CPIAUCSL  — BLS CPI-U, monthly, SA, 1947-present
  U-3             : UNRATE    — BLS civilian unemployment, monthly, SA, 1948-present
  Mortgage 30yr   : MORTGAGE30US — Freddie Mac PMMS, weekly, 1971-present
  Real wages      : COMPRNFB  — BLS Nonfarm Business Real Compensation/Hour, quarterly, 1947-present
  Consumer sent.  : UMCSENT   — U of Michigan Surveys of Consumers, monthly, 1952-present
"""

import os
import sys
import time
import requests
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"

START = "1947-01-01"   # pull max history; dropna() below sets the true start once all windows are warm
END   = "2025-12-31"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "economic_sentiment.csv")

SERIES = {
    "cpi"               : "CPIAUCSL",
    "unemployment_u3"   : "UNRATE",
    "mortgage_30yr"     : "MORTGAGE30US",
    "real_wages"        : "COMPRNFB",
    "consumer_sentiment": "UMCSENT",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def fetch(series_id: str) -> pd.Series:
    if not FRED_API_KEY:
        sys.exit(
            "ERROR: FRED_API_KEY not set.\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then run:  export FRED_API_KEY=your_key_here"
        )
    params = {
        "series_id"         : series_id,
        "api_key"           : FRED_API_KEY,
        "file_type"         : "json",
        "observation_start" : START,
        "observation_end"   : END,
    }
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        raise ValueError(f"No observations returned for {series_id}")

    s = pd.Series(
        {o["date"]: o["value"] for o in obs},
        name=series_id,
        dtype=object,
    )
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors="coerce")   # "." missing-value markers → NaN
    return s


def to_quarterly(s: pd.Series) -> pd.Series:
    """Resample any frequency to quarterly-end averages."""
    return s.resample("Q").mean()


def rolling_avg(s: pd.Series, years: int) -> pd.Series:
    """Mean of the preceding `years` years of quarterly data (current quarter excluded).
    Returns NaN until a full window of preceding data exists."""
    quarters = years * 4
    return s.rolling(window=quarters, min_periods=quarters).mean().shift(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Fetching data from FRED…")

    raw = {}
    for col, sid in SERIES.items():
        print(f"  {sid} ({col})")
        raw[col] = fetch(sid)
        time.sleep(0.5)

    print("Resampling to quarterly…")
    q = {}
    for col, s in raw.items():
        q[col] = to_quarterly(s)

    print("Computing CPI year-on-year change…")
    q["cpi"] = q["cpi"].pct_change(4) * 100   # % change vs same quarter prior year

    print("Computing rolling averages and deviations…")
    q["cpi_5yr_avg"]       = rolling_avg(q["cpi"],          5)
    q["cpi_10yr_avg"]      = rolling_avg(q["cpi"],         10)
    q["mortgage_5yr_avg"]  = rolling_avg(q["mortgage_30yr"], 5)
    q["mortgage_10yr_avg"] = rolling_avg(q["mortgage_30yr"], 10)

    q["cpi-cpi5"]          = q["cpi"] - q["cpi_5yr_avg"]
    q["cpi-cpi10"]         = q["cpi"] - q["cpi_10yr_avg"]
    q["mortgage-mortgage5"]  = q["mortgage_30yr"] - q["mortgage_5yr_avg"]
    q["mortgage-mortgage10"] = q["mortgage_30yr"] - q["mortgage_10yr_avg"]

    print("Building final dataset…")
    df = pd.DataFrame(q)

    # Trim to end date then drop any rows where a window wasn't yet full
    df = df.loc[:"2025-12-31"].dropna().copy()

    # Friendly quarter label
    df.index.name = "period_end"
    df.insert(0, "quarter", df.index.to_period("Q").astype(str))

    # Column order
    df = df[[
        "quarter",
        "real_wages",
        "unemployment_u3",
        "cpi",
        "cpi_5yr_avg",
        "cpi_10yr_avg",
        "cpi-cpi5",
        "cpi-cpi10",
        "mortgage_30yr",
        "mortgage_5yr_avg",
        "mortgage_10yr_avg",
        "mortgage-mortgage5",
        "mortgage-mortgage10",
        "consumer_sentiment",
    ]]

    df = df.round(4)

    out = os.path.abspath(OUTPUT_PATH)
    df.to_csv(out)
    print(f"\nSaved {len(df)} rows → {out}")
    print(df.head())


if __name__ == "__main__":
    main()
