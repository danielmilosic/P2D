import pandas as pd
from pathlib import Path
from datetime import datetime
import requests
from P2D.utils import spacecraft_ID

PRIMARY_BASE = Path(r'./reduced_data')
SECONDARY_BASE = Path(r'C:/Users/14milosi/DRaD/reduced_data')
GITHUB_BASE = "https://raw.githubusercontent.com/danielmilosic/DRaD/master/reduced_data"

def load(spacecraft, timerange, cadence=None):
    """
    Load spacecraft data between timerange. Tries local (secondary, primary), then GitHub.
    
    Parameters:
        spacecraft (str)
        timerange (str | datetime | Timestamp | tuple of str/datetime)
        cadence (str or None): e.g. '1H'

    Returns:
        pd.DataFrame
    """
    spacecraft = spacecraft_ID(spacecraft, ID_number=False).lower()
    start_date, end_date = normalize_timerange(timerange)
    dfs = []
    current = start_date

    while current <= end_date:
        local_path = find_existing_file(spacecraft, current)

        if not local_path:
            local_path = get_file_path(spacecraft, current, PRIMARY_BASE)
            try:
                download_from_github(spacecraft, current, local_path)
            except FileNotFoundError as e:
                print(f"[Warning] Skipping {current.strftime('%Y-%m')}: {e}")
                current = increment_month(current)
                continue

        df = pd.read_parquet(local_path)
        dfs.append(df)
        current = increment_month(current)

    if not dfs:
        print("[Info] No data was found for the requested timerange.")
        return pd.DataFrame()

    df_all = pd.concat(dfs)
    #df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    #df_all.set_index('timestamp', inplace=True)
    #df_all = df_all.sort_index()
    df_all = df_all[start_date:end_date]

    if cadence:
        df_all = df_all.resample(cadence).mean()

    return df_all.reset_index()

def normalize_timerange(timerange):
    """
    Normalize timerange input to (start_date, end_date) as datetime objects.
    Accepts:
        - single string (e.g., '2020-05' or '2020-05-15')
        - single datetime or pd.Timestamp
        - tuple/list of 2 elements
    """
    if isinstance(timerange, (str, datetime, pd.Timestamp)):
        start = pd.to_datetime(timerange)
        end = increment_month(start) - pd.Timedelta(seconds=1)  # Inclusive to end of month
    elif isinstance(timerange, (list, tuple)) and len(timerange) == 2:
        start = pd.to_datetime(timerange[0])
        end = pd.to_datetime(timerange[1])
        if start > end:
            raise ValueError("Start of timerange must be before end.")
    else:
        raise TypeError("Invalid timerange format. Use str, datetime, or tuple of two dates.")
    return start, end

def increment_month(date):
    """
    Return the first day of the next month.
    """
    if date.month == 12:
        return date.replace(year=date.year + 1, month=1, day=1)
    else:
        return date.replace(month=date.month + 1, day=1)

def find_existing_file(spacecraft, date):
    for base in [SECONDARY_BASE, PRIMARY_BASE]:
        path = get_file_path(spacecraft, date, base)
        if path.exists():
            return path
    return None

def get_file_path(spacecraft, date, base_dir):
    year = date.year
    month = f"{date.month:02d}"
    filename = f"{spacecraft}_data{year}-{month}.parquet"
    return base_dir / spacecraft / filename

def get_github_url(spacecraft, date):
    year = date.year
    month = f"{date.month:02d}"
    filename = f"{spacecraft}_data{year}-{month}.parquet"
    return f"{GITHUB_BASE}/{spacecraft}/{filename}"

def download_from_github(spacecraft, date, dest_path):
    url = get_github_url(spacecraft, date)
    print(f"[Info] Attempting download from {url}")
    response = requests.get(url)

    if response.status_code != 200:
        raise FileNotFoundError(f"GitHub file not available ({response.status_code})")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, 'wb') as f:
        f.write(response.content)

    print(f"[Success] Saved to {dest_path}")
