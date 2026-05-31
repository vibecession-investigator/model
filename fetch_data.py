"""
Fetch economic data from FRED (Federal Reserve Economic Data) and produce
a monthly CSV dataset starting from when all rolling windows are warm (~late 1980).

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


def to_monthly(s: pd.Series, quarterly: bool = False) -> pd.Series:
    """Resample to month-end. Weekly/daily → mean. Quarterly → forward-fill."""
    if quarterly:
        return s.resample("M").ffill()
    return s.resample("M").mean()


def rolling_avg(s: pd.Series, years: int) -> pd.Series:
    """Mean of the preceding `years` years of monthly data (current month excluded).
    Returns NaN until a full window of preceding data exists."""
    months = years * 12
    return s.rolling(window=months, min_periods=months).mean().shift(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Fetching data from FRED…")

    raw = {}
    for col, sid in SERIES.items():
        print(f"  {sid} ({col})")
        raw[col] = fetch(sid)
        time.sleep(0.5)

    print("Resampling to monthly…")
    m = {}
    for col, s in raw.items():
        m[col] = to_monthly(s, quarterly=(col == "real_wages"))

    print("Computing CPI year-on-year change…")
    m["cpi"] = m["cpi"].pct_change(12) * 100   # % change vs same month prior year

    print("Computing rolling averages and deviations…")
    m["cpi_5yr_avg"]       = rolling_avg(m["cpi"],          5)
    m["cpi_10yr_avg"]      = rolling_avg(m["cpi"],         10)
    m["mortgage_5yr_avg"]  = rolling_avg(m["mortgage_30yr"], 5)
    m["mortgage_10yr_avg"] = rolling_avg(m["mortgage_30yr"], 10)

    m["cpi-cpi5"]            = m["cpi"] - m["cpi_5yr_avg"]
    m["cpi-cpi10"]           = m["cpi"] - m["cpi_10yr_avg"]
    m["mortgage-mortgage5"]  = m["mortgage_30yr"] - m["mortgage_5yr_avg"]
    m["mortgage-mortgage10"] = m["mortgage_30yr"] - m["mortgage_10yr_avg"]

    print("Building final dataset…")
    df = pd.DataFrame(m)

    # Drop any rows where a window wasn't yet full
    df = df.loc[:"2025-12-31"].dropna().copy()

    df.index.name = "period_end"

    # Column order
    df = df[[
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
