import os, datetime as dt
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import requests

# Optional Streamlit import for caching when inside app
try:
    import streamlit as st
except Exception:
    st = None

# ---------- Conditional cache decorator ----------
from functools import lru_cache

def cache_auto(**st_kwargs):
    """
    Use Streamlit cache when inside a Streamlit run; otherwise lru_cache.
    Avoids noisy 'No runtime found' warnings.
    """
    def _wrap(fn):
        if st is not None:
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                if get_script_run_ctx() is not None:
                    return st.cache_data(**st_kwargs)(fn)
            except Exception:
                pass
        return lru_cache(maxsize=1)(fn)
    return _wrap

# ---------- Data loading / merging ----------
@cache_auto(show_spinner=False)
def load_and_merge_data() -> pd.DataFrame:
    data_dir = "data"
    req = ["yield_df.csv", "pesticides.csv", "rainfall.csv", "temp.csv"]
    for f in req:
        if not os.path.exists(os.path.join(data_dir, f)):
            raise FileNotFoundError(f"Missing data/{f}. Please place the CSVs as in your project.")
    yield_df = pd.read_csv(os.path.join(data_dir, "yield_df.csv"))
    pesticides = pd.read_csv(os.path.join(data_dir, "pesticides.csv"))
    rainfall = pd.read_csv(os.path.join(data_dir, "rainfall.csv"))
    temp = pd.read_csv(os.path.join(data_dir, "temp.csv"))

    rainfall = rainfall.rename(columns={' Area': 'Area'})
    temp = temp.rename(columns={'year': 'Year', 'country': 'Area'})
    pesticides = pesticides.rename(columns={'Value': 'pesticides_tonnes'})

    for df_ in [yield_df, rainfall, temp, pesticides]:
        if 'Area' in df_.columns:
            df_['Area'] = df_['Area'].astype(str).str.strip()
        if 'Item' in df_.columns:
            df_['Item'] = df_['Item'].astype(str).str.strip()

    df = yield_df.copy()
    if set(['Area','Year']).issubset(rainfall.columns):
        df = df.merge(rainfall[['Area','Year','average_rain_fall_mm_per_year']], on=['Area','Year'], how='left')
    if set(['Area','Year']).issubset(temp.columns):
        df = df.merge(temp[['Area','Year','avg_temp']], on=['Area','Year'], how='left')
    if set(['Area','Year']).issubset(pesticides.columns):
        df = df.merge(pesticides[['Area','Year','pesticides_tonnes']], on=['Area','Year'], how='left')

    for name in ["average_rain_fall_mm_per_year", "avg_temp", "pesticides_tonnes"]:
        x, y = f"{name}_x", f"{name}_y"
        if x in df.columns and y in df.columns:
            df[name] = df[x].fillna(df[y])
            df.drop(columns=[x, y], inplace=True, errors="ignore")

    num_cols = df.select_dtypes(include=['float64','int64','float32','int32']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df

# Compatibility alias used by older imports
def cache_data(df: pd.DataFrame) -> pd.DataFrame:
    return df

def safe_number(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def season_from_year(year: int) -> str:
    m = {1:"rabi",2:"rabi",3:"rabi",4:"zaid",5:"zaid",6:"zaid",7:"kharif",8:"kharif",9:"kharif",10:"rabi",11:"rabi",12:"rabi"}
    return m.get((year or 6) if isinstance(year, int) else 6, "kharif")

# ---------- Auto-location + weather ----------
def geolocate_ip() -> Tuple[Optional[float], Optional[float], str]:
    try:
        j = requests.get("https://ipapi.co/json/", timeout=4).json()
        return float(j.get("latitude")), float(j.get("longitude")), j.get("city") or "Your area"
    except Exception:
        return None, None, "Your area"

def fetch_forecast(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "timezone": "auto",
              "current": "temperature_2m,precipitation",
              "daily": "rain_sum,temperature_2m_max,temperature_2m_min"}
    return requests.get(url, params=params, timeout=6).json()

def fetch_historical_daily(lat: float, lon: float, start: dt.date, end: dt.date) -> Dict[str, Any]:
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {"parameters": "T2M,PRECTOTCORR","community": "AG","latitude": lat,"longitude": lon,
              "start": start.strftime("%Y%m%d"),"end": end.strftime("%Y%m%d"),"format": "JSON"}
    return requests.get(url, params=params, timeout=10).json()

def summarize_last_year_climate(lat: float, lon: float) -> Tuple[float, float]:
    end = dt.date.today(); start = end - dt.timedelta(days=365)
    try:
        js = fetch_historical_daily(lat, lon, start, end)
        param = (js or {}).get("properties", {}).get("parameter", {})
        t2m = param.get("T2M", {}); pre = param.get("PRECTOTCORR", {})
        annual_rain = float(sum(v for v in pre.values() if v is not None))
        avg_temp = float(sum(v for v in t2m.values() if v is not None) / max(1, len(t2m)))
    except Exception:
        annual_rain, avg_temp = 1000.0, 25.0
    return annual_rain, avg_temp

def forecast_next48h_mm(forecast: Dict[str, Any]) -> Optional[float]:
    try:
        return float(sum(forecast.get("daily", {}).get("rain_sum", [])[:2]))
    except Exception:
        return None
