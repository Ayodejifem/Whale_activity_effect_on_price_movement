# Whale Activity vs Price Movement (Coingecko + Dune)

A small Python CLI that fetches token price data from Coingecko and whale activity from Dune API, aligns the time series, and computes/plots correlations (including lead/lag cross-correlation).

## Setup

1. Python 3.10+ is recommended.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Export your Dune API key or create a `.env` file:

```bash
export DUNE_API_KEY=your_dune_api_key_here
```

## Usage

Basic example (hourly):

```bash
python whale_activity_analysis.py \
  --coin-id ethereum \
  --vs-currency usd \
  --days 90 \
  --dune-query-id 1234567 \
  --time-col block_time \
  --value-col amount_usd \
  --resample H \
  --aggregate sum \
  --save-csv
```

If your Dune query needs parameters, pass `--dune-params` as JSON. You can specify a simple object which will be converted to Dune parameters automatically:

```bash
python whale_activity_analysis.py \
  --coin-id uniswap \
  --days 180 \
  --dune-query-id 7654321 \
  --dune-params '{"start_time":"2024-01-01 00:00:00 UTC","end_time":"2024-06-30 23:59:59 UTC"}'
```

If you want to force fresh execution instead of using the latest stored result:

```bash
python whale_activity_analysis.py \
  --coin-id ethereum \
  --dune-query-id 1234567 \
  --dune-source execute \
  --dune-performance large
```

Outputs are written to `./outputs` by default:
- `<prefix>_overview.png`: Dual-axis plot of price and whale metric
- `<prefix>_xcorr.png`: Cross-correlation bar chart (positive lags = whale leads)
- `<prefix>_merged.csv`: Optional merged dataset when `--save-csv` is used

## Notes
- Coingecko `--days` can be a number or `max`.
- Choose `--resample H` (hourly) or `--resample D` (daily) to align series.
- `--value-col` should match your Dune query's output for the whale activity metric (e.g., `amount_usd`, `whale_volume_usd`, `whale_tx_count`).
- For best cross-correlation insights, ensure your Dune query returns a reasonably continuous time series at the chosen granularity.