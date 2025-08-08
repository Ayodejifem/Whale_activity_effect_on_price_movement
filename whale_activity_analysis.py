#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DUNE_API_BASE = "https://api.dune.com/api/v1"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def parse_dt(dt_str: str) -> datetime:
    # Accepts many formats; returns UTC-aware
    try:
        # Try ISO first
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        # Try common formats
        for fmt in [
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(dt_str, fmt)
                break
            except ValueError:
                continue
        else:
            raise
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_coingecko_market_chart(coin_id: str, vs_currency: str, days: str) -> pd.DataFrame:
    # days can be number or 'max'
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Coingecko error {r.status_code}: {r.text[:200]}")
    data = r.json()
    # Data contains 'prices', 'market_caps', 'total_volumes' arrays of [ms, value]
    df_price = pd.DataFrame(data.get("prices", []), columns=["timestamp_ms", "price"])
    df_mc = pd.DataFrame(data.get("market_caps", []), columns=["timestamp_ms", "market_cap"])
    df_vol = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp_ms", "volume"])

    df = df_price.merge(df_mc, on="timestamp_ms", how="outer").merge(df_vol, on="timestamp_ms", how="outer")
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])  # keep unified ts
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def dune_headers(api_key: str) -> Dict[str, str]:
    return {"x-dune-api-key": api_key}


def dune_get_latest_result(query_id: int, api_key: str, limit: Optional[int] = None, offset: Optional[int] = None,
                           columns: Optional[List[str]] = None, filters: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    if columns:
        params["columns"] = ",".join(columns)
    if filters:
        params["filters"] = filters
    url = f"{DUNE_API_BASE}/query/{query_id}/results"
    r = requests.get(url, headers=dune_headers(api_key), params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Dune latest results error {r.status_code}: {r.text[:200]}")
    return r.json()


def dune_execute_query(query_id: int, api_key: str, parameters: Optional[List[Dict[str, Any]]] = None,
                       performance: Optional[str] = None) -> str:
    url = f"{DUNE_API_BASE}/query/{query_id}/execute"
    body: Dict[str, Any] = {}
    if parameters:
        body["parameters"] = parameters
    if performance:
        body["performance"] = performance
    r = requests.post(url, headers=dune_headers(api_key), json=body, timeout=60)
    if r.status_code not in (200, 202):
        raise RuntimeError(f"Dune execute error {r.status_code}: {r.text[:200]}")
    data = r.json()
    exec_id = data.get("execution_id") or data.get("executionID")
    if not exec_id:
        raise RuntimeError(f"No execution_id in response: {data}")
    return exec_id


def dune_get_execution_status(execution_id: str, api_key: str) -> Dict[str, Any]:
    url = f"{DUNE_API_BASE}/execution/{execution_id}/status"
    r = requests.get(url, headers=dune_headers(api_key), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Dune status error {r.status_code}: {r.text[:200]}")
    return r.json()


def dune_get_execution_result(execution_id: str, api_key: str, limit: Optional[int] = None,
                              offset: Optional[int] = None, columns: Optional[List[str]] = None,
                              filters: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    if columns:
        params["columns"] = ",".join(columns)
    if filters:
        params["filters"] = filters
    url = f"{DUNE_API_BASE}/execution/{execution_id}/results"
    r = requests.get(url, headers=dune_headers(api_key), params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Dune execution results error {r.status_code}: {r.text[:200]}")
    return r.json()


def poll_dune_until_done(execution_id: str, api_key: str, timeout_sec: int = 600, poll_every_sec: int = 5) -> Dict[str, Any]:
    start = time.time()
    while True:
        status = dune_get_execution_status(execution_id, api_key)
        state = status.get("state") or status.get("execution_state")
        is_finished = status.get("is_execution_finished")
        if state in {"QUERY_STATE_COMPLETED", "QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"} or is_finished:
            return status
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Dune execution timed out after {timeout_sec}s. Last status: {status}")
        time.sleep(poll_every_sec)


def rows_to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    result = payload.get("result") or {}
    rows = result.get("rows") or []
    df = pd.DataFrame(rows)
    return df


def normalize_time_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found in Dune result columns: {list(df.columns)}")
    # Parse using pandas to_datetime, handle UTC suffix
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return df


def resample_series(df: pd.DataFrame, time_col: str, value_col: str, freq: str, how: str = "sum") -> pd.DataFrame:
    df = df[[time_col, value_col]].copy()
    df = df.set_index(time_col).sort_index()
    if how == "sum":
        series = df[value_col].resample(freq).sum()
    elif how == "mean":
        series = df[value_col].resample(freq).mean()
    elif how == "count":
        series = df[value_col].resample(freq).count()
    else:
        raise ValueError("how must be one of 'sum', 'mean', 'count'")
    out = series.to_frame(name=value_col).reset_index()
    return out


def compute_returns(price_df: pd.DataFrame, freq: str) -> pd.Series:
    price_df = price_df.set_index("timestamp").sort_index()
    # Resample close price per bin
    close_price = price_df["price"].resample(freq).last().dropna()
    returns = close_price.pct_change().dropna()
    returns.name = "returns"
    return returns


def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    # Align index
    combined = pd.concat([x, y], axis=1, join="inner").dropna()
    x_vals = combined.iloc[:, 0].values - combined.iloc[:, 0].values.mean()
    y_vals = combined.iloc[:, 1].values - combined.iloc[:, 1].values.mean()
    corrs = []
    lags = np.arange(-max_lag, max_lag + 1)
    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(x_vals[-lag:], y_vals[:len(y_vals)+lag])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x_vals[:len(x_vals)-lag], y_vals[lag:])[0, 1]
        else:
            corr = np.corrcoef(x_vals, y_vals)[0, 1]
        corrs.append(corr)
    return lags, np.array(corrs)


def plot_overview(price: pd.DataFrame, whale: pd.DataFrame, time_col: str, value_col: str, out_path: str,
                  title: str) -> None:
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(price["timestamp"], price["price"], color="tab:blue", label="Price")
    ax1.set_ylabel("Price", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(whale[time_col], whale[value_col], color="tab:orange", label=value_col)
    ax2.set_ylabel(value_col, color="tab:orange")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax1.set_title(title)
    ax1.set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_xcorr(lags: np.ndarray, corrs: np.ndarray, out_path: str, title: str) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(lags, corrs, color="tab:green")
    ax.set_title(title)
    ax.set_xlabel("Lag (periods; positive means whale leads)")
    ax.set_ylabel("Correlation")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_dune_params(json_str: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not json_str:
        return None
    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse --dune-params JSON: {e}")

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # Convert simple dict to Dune parameters list (type defaults to 'text')
        out: List[Dict[str, Any]] = []
        for k, v in raw.items():
            p: Dict[str, Any] = {"key": k, "value": v}
            out.append(p)
        return out
    raise ValueError("--dune-params must be a JSON object or array")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze whale activity effect on price using Coingecko and Dune data.")
    parser.add_argument("--coin-id", required=True, help="Coingecko coin id (e.g., ethereum, bitcoin, uniswap)")
    parser.add_argument("--vs-currency", default="usd", help="Fiat currency for Coingecko (default: usd)")
    parser.add_argument("--days", default="90", help="Days for Coingecko data (e.g., 30, 90, 365, max)")

    parser.add_argument("--dune-query-id", type=int, required=True, help="Dune query ID returning whale activity time series")
    parser.add_argument("--dune-api-key", default=os.getenv("DUNE_API_KEY"), help="Dune API key (or set DUNE_API_KEY env)")
    parser.add_argument("--dune-params", default=None, help="JSON for Dune query parameters (dict or list)")
    parser.add_argument("--dune-performance", default=None, help="Optional Dune performance tier: small|medium|large")
    parser.add_argument("--dune-source", choices=["latest", "execute"], default="latest",
                        help="Fetch latest stored result or execute and fetch (default: latest)")

    parser.add_argument("--time-col", default="block_time", help="Dune time column name (default: block_time)")
    parser.add_argument("--value-col", default="amount_usd", help="Dune value column for whale activity (default: amount_usd)")
    parser.add_argument("--resample", choices=["H", "D"], default="H", help="Resample frequency: H=hourly, D=daily")
    parser.add_argument("--aggregate", choices=["sum", "mean", "count"], default="sum",
                        help="Aggregation for whale metric when resampling (default: sum)")

    parser.add_argument("--max-lag", type=int, default=48, help="Max lag for cross-correlation in periods (default: 48)")

    parser.add_argument("--out-dir", default="./outputs", help="Directory to write outputs")
    parser.add_argument("--prefix", default=None, help="Filename prefix (default: coin id)")
    parser.add_argument("--save-csv", action="store_true", help="Save merged dataset to CSV")

    args = parser.parse_args(argv)

    if not args.dune_api_key:
        print("Error: Dune API key not provided. Use --dune-api-key or set DUNE_API_KEY.", file=sys.stderr)
        return 2

    ensure_output_dir(args.out_dir)
    prefix = args.prefix or args.coin_id

    # Fetch price data
    price_df = fetch_coingecko_market_chart(args.coin_id, args.vs_currency, args.days)

    # Fetch Dune data
    dune_params = parse_dune_params(args.dune_params)
    df_dune: pd.DataFrame
    if args.dune_source == "latest":
        payload = dune_get_latest_result(args.dune_query_id, args.dune_api_key)
        df_dune = rows_to_dataframe(payload)
    else:
        exec_id = dune_execute_query(args.dune_query_id, args.dune_api_key, parameters=dune_params,
                                     performance=args.dune_performance)
        _ = poll_dune_until_done(exec_id, args.dune_api_key)
        payload = dune_get_execution_result(exec_id, args.dune_api_key)
        df_dune = rows_to_dataframe(payload)

    if df_dune.empty:
        raise RuntimeError("Dune returned no rows. Ensure your query id and permissions are correct.")

    df_dune = normalize_time_column(df_dune, args.time_col)
    if args.value_col not in df_dune.columns:
        raise KeyError(f"Value column '{args.value_col}' not in Dune result. Available: {list(df_dune.columns)}")

    # Resample whale data to requested frequency
    whale_ts = resample_series(df_dune, args.time_col, args.value_col, args.resample, how=args.aggregate)

    # Align price to same frequency and compute returns
    returns = compute_returns(price_df, args.resample)

    # Prepare merged
    merged = whale_ts.set_index(args.time_col).join(returns, how="inner")
    merged = merged.dropna(subset=[args.value_col, "returns"])  # ensure both present

    if merged.empty:
        raise RuntimeError("After alignment, no overlapping data between price and whale metrics.")

    # Basic correlation
    pearson_corr = merged[args.value_col].corr(merged["returns"])  # contemporaneous

    # Cross-correlation across lags
    lags, corrs = cross_correlation(merged[args.value_col], merged["returns"], max_lag=args.max_lag)
    best_idx = int(np.nanargmax(np.abs(corrs)))
    best_lag = int(lags[best_idx])
    best_corr = float(corrs[best_idx])

    # Outputs
    overview_path = os.path.join(args.out_dir, f"{prefix}_overview.png")
    xcorr_path = os.path.join(args.out_dir, f"{prefix}_xcorr.png")

    plot_overview(price_df, whale_ts, args.time_col, args.value_col, overview_path,
                  title=f"{args.coin_id} price vs {args.value_col} ({args.resample})")
    plot_xcorr(lags, corrs, xcorr_path,
               title=f"Cross-correlation: {args.value_col} vs returns (max |corr| at lag {best_lag}: {best_corr:.3f})")

    if args.save_csv:
        csv_path = os.path.join(args.out_dir, f"{prefix}_merged.csv")
        merged.reset_index().to_csv(csv_path, index=False)

    # Console summary
    print(f"Rows aligned: {len(merged)}")
    print(f"Contemporaneous Pearson corr (whale metric vs returns): {pearson_corr:.4f}")
    print(f"Best lag: {best_lag} periods; correlation: {best_corr:.4f}")
    print(f"Saved overview plot to: {overview_path}")
    print(f"Saved xcorr plot to: {xcorr_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())