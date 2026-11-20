"""Data acquisition and cleaning.

1. Dengue cases: `srilanka_weekly_data` from the R package denguedatahub
   (Thiyanga S. Talagala), downloaded as an .rda file from GitHub and read with pyreadr.
2. Weather: Open-Meteo Historical Weather API (ERA5 reanalysis), daily values per
   district centre, cached to data/raw/weather.csv so the API is called only once.

Both raw files are saved unchanged in data/raw/; cleaned versions go to data/processed/.
"""
import time

import pandas as pd
import requests

from .config import DATA_PROCESSED, DATA_RAW, DENGUE_RAW_CSV, DISTRICT_COORDS, WEATHER_RAW_CSV

DENGUE_RDA_URL = ("https://raw.githubusercontent.com/thiyangt/denguedatahub/main/"
                  "data/srilanka_weekly_data.rda")
DENGUE_RDA_PATH = DATA_RAW / "srilanka_weekly_data.rda"
DENGUE_CLEAN_CSV = DATA_PROCESSED / "dengue_clean.csv"
WEATHER_WEEKLY_CSV = DATA_PROCESSED / "weather_weekly.csv"

# Spelling variants in the source -> standard district / RDHS names used in config.
NAME_FIXES = {
    "Hambanthota": "Hambantota",
    "NuwaraEliya": "Nuwara Eliya",
    "Kalmune": "Kalmunai",
}


# ============================================================ dengue
def download_dengue(force: bool = False) -> pd.DataFrame:
    """Download the denguedatahub .rda file and export it unchanged as data/raw/dengue.csv."""
    import pyreadr  # only needed for data collection, not by the dashboard

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if force or not DENGUE_RDA_PATH.exists():
        resp = requests.get(DENGUE_RDA_URL, timeout=60)
        resp.raise_for_status()
        DENGUE_RDA_PATH.write_bytes(resp.content)
    raw = pyreadr.read_r(str(DENGUE_RDA_PATH))["srilanka_weekly_data"]
    raw.to_csv(DENGUE_RAW_CSV, index=False)
    return raw


def clean_dengue(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise names, parse dates and rebuild a reliable (year, week) calendar.

    Why rebuild year/week? In the source, labels around New Year are inconsistent
    (e.g. the week 23-29 Dec 2023 is labelled "2024 week 1", and 20-26 Dec 2025 is
    "2026 week 53"), and from 2026 reporting weeks start on Monday instead of Saturday.
    The `start.date` column is consistent, so we treat it as the truth and derive:
        mid_date = start_date + 3 days         (middle of the reporting week)
        year     = mid_date.year
        week     = (day-of-year of mid_date - 1) // 7 + 1    -> 1..53
    """
    df = raw.rename(columns={"start.date": "start_date", "end.date": "end_date"}).copy()
    # Dates appear in two formats ("12/23/2006" and "2026-09-07"); format="mixed" handles both.
    df["start_date"] = pd.to_datetime(df["start_date"], format="mixed")
    df["end_date"] = pd.to_datetime(df["end_date"], format="mixed")
    df["district"] = df["district"].str.strip().replace(NAME_FIXES)

    unknown = set(df["district"]) - set(DISTRICT_COORDS)
    assert not unknown, f"Unmapped district names: {unknown}"

    mid = df["start_date"] + pd.Timedelta(days=3)
    df["year"] = mid.dt.year
    df["week"] = (mid.dt.dayofyear - 1) // 7 + 1
    df = (df[["district", "start_date", "end_date", "year", "week", "cases"]]
          .sort_values(["district", "start_date"]).reset_index(drop=True))
    assert not df.duplicated(["district", "start_date"]).any(), "duplicate district-weeks"
    return flag_missing_reports(df)


# A week is a national reporting gap when at least this many of the 26 regions report
# exactly 0 AND that is far above the surrounding weeks (see flag_missing_reports).
GAP_WEEK_MIN_ZERO_REGIONS = 20
GAP_WEEK_JUMP = 10          # zero-count must exceed the median of +-3 neighbouring weeks by this much


def flag_missing_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Mark zeros that are really *missing reports*, not true zero-case weeks.

    Two patterns found in EDA:
      1. A region reports 0 for an ENTIRE calendar year while reporting cases in other
         years (Kilinochchi & Mullaitivu 2007-2009, during the final phase of the civil war;
         Monaragala 2020).
      2. A single week where >= 20 of 26 regions report 0 while the surrounding weeks have
         only a handful (10 Sep 2022: 23 regions vs 1-6 in neighbouring weeks) - a national
         reporting gap. The "jump vs neighbours" condition avoids flagging periods like
         mid-2012 where many small regions genuinely reported 0 week after week.
    These rows get cases = NaN and missing_report = True. We do NOT impute values
    (that would be inventing data); feature rows touching them are dropped later.
    """
    df = df.copy()
    full_year = df.groupby(["district", "year"])["cases"].transform(lambda s: len(s) >= 50 and s.sum() == 0)

    zeros = df.groupby("start_date")["cases"].apply(lambda s: (s == 0).sum()).sort_index()
    neighbours = pd.concat([zeros.shift(k) for k in (-3, -2, -1, 1, 2, 3)], axis=1).median(axis=1)
    gap_dates = zeros.index[(zeros >= GAP_WEEK_MIN_ZERO_REGIONS) & (zeros - neighbours >= GAP_WEEK_JUMP)]
    gap_week = df["start_date"].isin(gap_dates) & (df["cases"] == 0)
    df["missing_report"] = full_year.astype(bool) | gap_week
    df["cases"] = df["cases"].astype(float).mask(df["missing_report"])
    return df


def load_dengue_clean() -> pd.DataFrame:
    return pd.read_csv(DENGUE_CLEAN_CSV, parse_dates=["start_date", "end_date"])


# ============================================================ weather
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = ["precipitation_sum", "temperature_2m_mean", "relative_humidity_2m_mean"]


def _fetch_point(lat, lon, start, end, retries=6):
    """One Open-Meteo archive request; retries with back-off on rate limits (HTTP 429)."""
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "daily": ",".join(DAILY_VARS), "timezone": "Asia/Colombo"}
    for attempt in range(retries):
        r = requests.get(ARCHIVE_URL, params=params, timeout=120)
        if r.status_code == 200:
            return pd.DataFrame(r.json()["daily"])
        if r.status_code == 429:           # rate limited: wait and retry
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError("Open-Meteo rate limit not cleared after retries")


def download_weather(start: str, end: str, force: bool = False) -> pd.DataFrame | None:
    """Daily weather for every district centre, cached to data/raw/weather.csv.

    Returns None (and the pipeline continues without weather) if the API is unreachable.
    Districts sharing coordinates (Kalmunai = Ampara) are fetched once and copied.
    """
    if WEATHER_RAW_CSV.exists() and not force:
        return pd.read_csv(WEATHER_RAW_CSV, parse_dates=["date"])

    frames, cache = [], {}
    try:
        for district, (lat, lon) in DISTRICT_COORDS.items():
            if (lat, lon) not in cache:
                cache[(lat, lon)] = _fetch_point(lat, lon, start, end)
                time.sleep(2)              # be polite to the free API
            d = cache[(lat, lon)].copy()
            d.insert(0, "district", district)
            frames.append(d)
            print(f"  weather ok: {district}")
    except Exception as exc:              # network down, API changed, ...
        print(f"Weather download failed ({exc}); continuing WITHOUT weather features.")
        return None

    w = pd.concat(frames, ignore_index=True).rename(columns={
        "time": "date", "precipitation_sum": "rain_mm",
        "temperature_2m_mean": "temp_c", "relative_humidity_2m_mean": "humidity_pct"})
    w["date"] = pd.to_datetime(w["date"])
    w.to_csv(WEATHER_RAW_CSV, index=False)
    return w


def weekly_weather(daily: pd.DataFrame, dengue: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather onto the exact dengue reporting weeks.

    Each day is assigned to the reporting week (start_date..end_date) that contains it,
    using merge_asof per district. Rain is summed; temperature and humidity are averaged.
    """
    weeks = dengue[["district", "start_date", "end_date"]].sort_values("start_date")
    daily = daily.sort_values("date")
    m = pd.merge_asof(daily, weeks, left_on="date", right_on="start_date",
                      by="district", direction="backward")
    m = m[m["date"] <= m["end_date"]]      # drop days that fall in a reporting gap
    wk = (m.groupby(["district", "start_date"])
           .agg(rain_mm=("rain_mm", "sum"), temp_c=("temp_c", "mean"),
                humidity_pct=("humidity_pct", "mean"), n_days=("date", "size"))
           .reset_index())
    return wk


def build_clean_datasets(with_weather: bool = True) -> pd.DataFrame:
    """End-to-end: download + clean dengue, add weekly weather, save to data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    dengue = clean_dengue(download_dengue())
    dengue.to_csv(DENGUE_CLEAN_CSV, index=False)

    if with_weather:
        start = dengue["start_date"].min().strftime("%Y-%m-%d")
        end = dengue["end_date"].max().strftime("%Y-%m-%d")
        daily = download_weather(start, end)
        if daily is not None:
            wk = weekly_weather(daily, dengue)
            wk.to_csv(WEATHER_WEEKLY_CSV, index=False)
    return dengue


def load_weather_weekly() -> pd.DataFrame | None:
    if not WEATHER_WEEKLY_CSV.exists():
        return None
    return pd.read_csv(WEATHER_WEEKLY_CSV, parse_dates=["start_date"])


def national_weekly(dengue: pd.DataFrame) -> pd.Series:
    """National total cases per reporting week (indexed by start_date).

    Weeks flagged as national reporting gaps become NaN (a sum over mostly-missing
    reports would look like a fake collapse). Weeks where only one region is missing
    (e.g. Monaragala 2020) keep the sum of the regions that did report.
    """
    g = dengue.groupby("start_date")
    total = g["cases"].sum(min_count=1)
    n_missing = g["missing_report"].sum()
    return total.mask(n_missing >= GAP_WEEK_MIN_ZERO_REGIONS - 5).astype(float)  # >=15 regions missing
