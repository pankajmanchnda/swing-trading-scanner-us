import math
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


CAPITAL_USD = 100_000
RISK_PER_TRADE = 0.01
MAX_ALLOCATION = 0.20
RR_TARGET = 2.5
TOP_N = 3
EARNINGS_BLACKOUT_DAYS = 8

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "CRM", "ADBE", "INTC", "QCOM", "MU", "ORCL", "IBM", "NOW",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK", "SCHW",
    "UNH", "LLY", "JNJ", "MRK", "ABBV", "PFE", "TMO", "ISRG", "ABT", "DHR",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "HD", "LOW", "MCD", "SBUX", "NKE", "TGT", "WMT",
    "BA", "CAT", "DE", "GE", "LMT", "RTX", "HON", "UPS",
    "SPY", "QQQ"
]

BENCHMARK = "SPY"
OUTPUT_HTML = "index.html"
PERFORMANCE_LOG = "performance_log_us.csv"
IST = timezone(timedelta(hours=5, minutes=30))
ETF_SYMBOLS = {"SPY", "QQQ", "DIA", "IWM"}

MODES = {
    "swing": {
        "label": "Swing",
        "description": "Daily candle swing-trade view",
        "period": "6mo",
        "interval": "1d",
        "bias_change_label": "5-Day Change",
        "bias_lookback_bars": 6,
        "rs_lookback_bars": 22,
    },
    "intraday": {
        "label": "Intraday",
        "description": "15-minute candle active-trade view",
        "period": "30d",
        "interval": "15m",
        "bias_change_label": "Recent Change",
        "bias_lookback_bars": 22,
        "rs_lookback_bars": 22,
    },
}


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def get_earnings_risk(ticker):
    """
    Returns earnings-risk status for a ticker.

    Logic:
    - If earnings date is within the next EARNINGS_BLACKOUT_DAYS, skip the stock.
    - If earnings date is unavailable, do not block the stock, but mark it as unknown.
    - ETFs are not blocked for earnings.
    """
    if ticker in ETF_SYMBOLS:
        return {
            "skip": False,
            "status": "ETF - earnings not applicable",
            "date": "",
        }

    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        earnings_date = None

        if calendar is None:
            return {"skip": False, "status": "Earnings date unavailable", "date": ""}

        if isinstance(calendar, dict):
            raw_date = calendar.get("Earnings Date") or calendar.get("EarningsDate")
            earnings_date = raw_date[0] if isinstance(raw_date, list) and len(raw_date) > 0 else raw_date

        elif isinstance(calendar, pd.DataFrame):
            if "Earnings Date" in calendar.index:
                raw_date = calendar.loc["Earnings Date"].values[0]
                earnings_date = raw_date[0] if isinstance(raw_date, list) and len(raw_date) > 0 else raw_date

        if earnings_date is None or pd.isna(earnings_date):
            return {"skip": False, "status": "Earnings date unavailable", "date": ""}

        earnings_date = pd.to_datetime(earnings_date).date()
        today = datetime.now(timezone.utc).date()
        days_until = (earnings_date - today).days

        if 0 <= days_until <= EARNINGS_BLACKOUT_DAYS:
            return {
                "skip": True,
                "status": f"Skipped: earnings in {days_until} days",
                "date": earnings_date.isoformat(),
            }

        return {"skip": False, "status": "Clear", "date": earnings_date.isoformat()}

    except Exception:
        return {"skip": False, "status": "Earnings check unavailable", "date": ""}


def get_single_df(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return pd.DataFrame()
        df = data[ticker].copy()
    else:
        df = data.copy()

    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], how="any")


def benchmark_bias(bench_df, config):
    b = bench_df.copy()
    b["EMA20"] = ema(b["Close"], 20)
    b["EMA50"] = ema(b["Close"], 50)

    close = float(b["Close"].iloc[-1])
    ema20 = float(b["EMA20"].iloc[-1])
    ema50 = float(b["EMA50"].iloc[-1])

    lookback = int(config["bias_lookback_bars"])
    if len(b) >= lookback:
        recent_change = ((b["Close"].iloc[-1] / b["Close"].iloc[-lookback]) - 1) * 100
    else:
        recent_change = 0

    if close > ema20 > ema50 and recent_change > 0:
        bias = "Bullish"
        message = "US market trend is supportive. Scanner will only show BUY setups."
    elif close < ema20 < ema50 and recent_change < 0:
        bias = "Bearish"
        message = "US market trend is weak. Scanner will only show SELL setups."
    else:
        bias = "Mixed"
        message = "US market is mixed. Scanner will only show very high-conviction setups."

    return {
        "bias": bias,
        "message": message,
        "close": round(close, 2),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "recent_change": round(float(recent_change), 2),
        "change_label": config["bias_change_label"],
    }


def score_ticker(df, bench_df, ticker, config):
    if len(df) < 80:
        return None

    d = df.copy()
    d["EMA20"] = ema(d["Close"], 20)
    d["EMA50"] = ema(d["Close"], 50)
    d["ATR14"] = atr(d, 14)
    d["RSI14"] = rsi(d["Close"], 14)
    d["VolAvg20"] = d["Volume"].rolling(20).mean()

    close = float(d["Close"].iloc[-1])
    ema20 = float(d["EMA20"].iloc[-1])
    ema50 = float(d["EMA50"].iloc[-1])
    atr14 = float(d["ATR14"].iloc[-1])
    rsi14 = float(d["RSI14"].iloc[-1])
    vol = float(d["Volume"].iloc[-1])
    vol_avg = float(d["VolAvg20"].iloc[-1])

    if not all(np.isfinite(x) for x in [close, ema20, ema50, atr14, rsi14, vol_avg]):
        return None

    vol_ratio = vol / vol_avg if vol_avg > 0 else np.nan
    atr_pct = (atr14 / close) * 100 if close > 0 else np.nan

    if not np.isfinite(vol_ratio) or not np.isfinite(atr_pct):
        return None

    rs_lookback = int(config["rs_lookback_bars"])
    if len(d) < rs_lookback or len(bench_df) < rs_lookback:
        return None

    stock_change = ((d["Close"].iloc[-1] / d["Close"].iloc[-rs_lookback]) - 1) * 100
    bench_change = ((bench_df["Close"].iloc[-1] / bench_df["Close"].iloc[-rs_lookback]) - 1) * 100
    rs_vs_bench = float(stock_change - bench_change)

    if not np.isfinite(rs_vs_bench):
        return None

    uptrend = close > ema20 > ema50
    downtrend = close < ema20 < ema50

    signal = None
    score = 50
    notes = []

    if uptrend and 50 <= rsi14 <= 70:
        signal = "BUY"
        score += 18
        notes.append("Uptrend")
    elif downtrend and 30 <= rsi14 <= 50:
        signal = "SELL"
        score += 18
        notes.append("Downtrend")
    else:
        return None

    if signal == "BUY" and rs_vs_bench > 3:
        score += 12
        notes.append("Strong relative strength")
    elif signal == "SELL" and rs_vs_bench < -3:
        score += 12
        notes.append("Weak relative strength")
    else:
        score += 5
        notes.append("Acceptable relative strength")

    if 1.5 <= atr_pct <= 5.5:
        score += 8
        notes.append("Tradable volatility")
    elif atr_pct > 7:
        score -= 8
        notes.append("High volatility")

    if vol_ratio >= 1.25:
        score += 7
        notes.append("Strong volume")
    elif vol_ratio >= 0.9:
        score += 3
        notes.append("Normal volume")

    if signal == "BUY":
        entry = max(close, d["High"].iloc[-1] * 1.002)
        stop = entry - 1.35 * atr14
        target = entry + RR_TARGET * (entry - stop)
        entry_rule = "Enter only above trigger"
    else:
        entry = min(close, d["Low"].iloc[-1] * 0.998)
        stop = entry + 1.35 * atr14
        target = entry - RR_TARGET * (stop - entry)
        entry_rule = "Enter only below trigger"

    risk_per_share = abs(entry - stop)
    max_risk_dollars = CAPITAL_USD * RISK_PER_TRADE
    qty_by_risk = math.floor(max_risk_dollars / risk_per_share) if risk_per_share > 0 else 0
    qty_by_allocation = math.floor((CAPITAL_USD * MAX_ALLOCATION) / entry) if entry > 0 else 0
    qty = max(0, min(qty_by_risk, qty_by_allocation))
    trade_value = qty * entry

    trigger_distance_pct = abs(entry - close) / close * 100

    # US-market quality filters.
    if trigger_distance_pct > 2.0:
        return None

    if vol_ratio < 0.90:
        return None

    if signal == "BUY" and rs_vs_bench < 0.5:
        return None

    if signal == "SELL" and rs_vs_bench > -0.5:
        return None

    if atr_pct > 6.5:
        return None

    # Calibrated scoring thresholds.
    if score >= 88:
        priority = "Highest Priority"
        grade = "A"
    elif score >= 83:
        priority = "Medium Priority"
        grade = "B+"
    elif score >= 78:
        priority = "Low Priority"
        grade = "B"
    else:
        return None

    return {
        "Priority": priority,
        "Stock": ticker,
        "Signal": signal,
        "Conviction": int(score),
        "Grade": grade,
        "Close": round(close, 2),
        "Entry": round(entry, 2),
        "Stop": round(stop, 2),
        "Target": round(target, 2),
        "RR": RR_TARGET,
        "RSI": round(rsi14, 1),
        "ATR%": f"{round(atr_pct, 2)}%",
        "Vol Ratio": round(vol_ratio, 2),
        "RS vs SPY": round(rs_vs_bench, 2),
        "Trigger Distance": f"{round(trigger_distance_pct, 2)}%",
        "Earnings": "Clear",
        "Earnings Date": "",
        "Qty": qty,
        "Trade Value": f"${round(trade_value, 0):,.0f}",
        "Entry Rule": entry_rule,
        "Notes": " · ".join(notes),
    }


def update_performance_log(mode_key, top3):
    log_path = Path(PERFORMANCE_LOG)
    today = datetime.now(timezone.utc).date().isoformat()
    mode_label = MODES[mode_key]["label"]

    columns = [
        "Date", "Mode", "Rank", "Stock", "Signal", "Conviction", "Entry", "Stop",
        "Target", "RR", "Status", "Result", "R", "Days", "Notes"
    ]

    if log_path.exists():
        log = pd.read_csv(log_path)
        if "Mode" not in log.columns:
            log["Mode"] = "Legacy"
        for col in columns:
            if col not in log.columns:
                log[col] = ""
        log = log[columns]
    else:
        log = pd.DataFrame(columns=columns)

    existing = set(zip(log.get("Date", []), log.get("Mode", []), log.get("Rank", []), log.get("Stock", [])))

    new_rows = []
    for rank, row in enumerate(top3, start=1):
        key = (today, mode_label, rank, row["Stock"])
        if key not in existing:
            new_rows.append({
                "Date": today,
                "Mode": mode_label,
                "Rank": rank,
                "Stock": row["Stock"],
                "Signal": row["Signal"],
                "Conviction": row["Conviction"],
                "Entry": row["Entry"],
                "Stop": row["Stop"],
                "Target": row["Target"],
                "RR": row["RR"],
                "Status": "Pending Trigger",
                "Result": "",
                "R": "",
                "Days": 0,
                "Notes": f"Top 3 {mode_label} scanner pick",
            })

    if new_rows:
        log = pd.concat([pd.DataFrame(new_rows), log], ignore_index=True)

    log.to_csv(log_path, index=False)
    return log[log["Mode"] == mode_label].head(60)


def run_scan(mode_key):
    config = MODES[mode_key]
    tickers = sorted(set(UNIVERSE + [BENCHMARK]))

    print(f"Downloading {config['label']} data: period={config['period']}, interval={config['interval']}...")
    data = yf.download(
        tickers=tickers,
        period=config["period"],
        interval=config["interval"],
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    bench_df = get_single_df(data, BENCHMARK)
    if bench_df.empty:
        raise RuntimeError(f"{BENCHMARK} benchmark data could not be downloaded for {config['label']} mode.")

    market = benchmark_bias(bench_df, config)
    rows = []

    for ticker in sorted(set(UNIVERSE)):
        if ticker == BENCHMARK:
            continue

        earnings = get_earnings_risk(ticker)
        if earnings["skip"]:
            continue

        df = get_single_df(data, ticker)
        result = score_ticker(df, bench_df, ticker, config)

        if result:
            # Do not allow trades against the broad US market trend.
            if market["bias"] == "Bullish" and result["Signal"] != "BUY":
                continue

            if market["bias"] == "Bearish" and result["Signal"] != "SELL":
                continue

            if market["bias"] == "Mixed" and result["Conviction"] < 90:
                continue

            result["Earnings"] = earnings["status"]
            result["Earnings Date"] = earnings["date"]
            rows.append(result)

    rows = sorted(rows, key=lambda x: x["Conviction"], reverse=True)
    perf_log = update_performance_log(mode_key, rows[:TOP_N])

    print(f"{config['label']} mode produced {len(rows)} candidates.")
    return {"mode_key": mode_key, "config": config, "market": market, "rows": rows, "perf_log": perf_log}


def build_cards(rows):
    cards = ""
    for r in rows[:TOP_N]:
        cards += f"""
        <div class="setup">
          <h3>{r['Stock']} — {r['Signal']}</h3>
          <p><b>{r['Priority']}</b> · Score {r['Conviction']}</p>
          <div class="grid">
            <div>Entry<br><b>{r['Entry']}</b></div>
            <div>Stop<br><b>{r['Stop']}</b></div>
            <div>Target<br><b>{r['Target']}</b></div>
            <div>Qty<br><b>{r['Qty']}</b></div>
            <div>Value<br><b>{r['Trade Value']}</b></div>
            <div>RR<br><b>{r['RR']}</b></div>
          </div>
          <p>{r['Entry Rule']} · {r['Notes']}</p>
        </div>
        """
    return cards if cards else "<p>No qualifying top setups for this mode.</p>"


def build_mode_block(scan):
    rows = scan["rows"]
    market = scan["market"]
    config = scan["config"]
    mode_key = scan["mode_key"]
    active_class = " active" if mode_key == "intraday" else ""

    total = len(rows)
    high = sum(r["Priority"] == "Highest Priority" for r in rows)
    medium = sum(r["Priority"] == "Medium Priority" for r in rows)
    low = sum(r["Priority"] == "Low Priority" for r in rows)
    best = max([r["Conviction"] for r in rows], default=0)

    table_html = (
        pd.DataFrame(rows).to_html(index=False, classes="data-table", escape=False)
        if rows
        else "<p>No qualifying setups in this mode.</p>"
    )

    perf_log = scan["perf_log"]
    perf_html = (
        perf_log.to_html(index=False, classes="data-table", escape=False)
        if not perf_log.empty
        else "<p>No performance records yet for this mode.</p>"
    )

    cards = build_cards(rows)

    return f"""
<section id="panel-{mode_key}" class="mode-panel{active_class}">
  <div class="mode-heading">
    <div>
      <h2>{config['label']} Mode</h2>
      <p>{config['description']} · period {config['period']} · interval {config['interval']}</p>
    </div>
  </div>

  <div class="stats">
    <div class="stat"><span>Total Candidates</span><b>{total}</b></div>
    <div class="stat"><span>Highest Priority</span><b>{high}</b></div>
    <div class="stat"><span>Medium Priority</span><b>{medium}</b></div>
    <div class="stat"><span>Low Priority</span><b>{low}</b></div>
    <div class="stat"><span>Best Conviction</span><b>{best}</b></div>
  </div>

  <div class="market">
    <h2>SPY Bias: {market['bias']}</h2>
    <p>{market['message']}</p>
    <div class="market-grid">
      <div>SPY Close<br><b>{market['close']}</b></div>
      <div>EMA 20<br><b>{market['ema20']}</b></div>
      <div>EMA 50<br><b>{market['ema50']}</b></div>
      <div>{market['change_label']}<br><b>{market['recent_change']}%</b></div>
    </div>
  </div>

  <h2>Top 3 Setups</h2>
  <div class="setup-wrap">{cards}</div>

  <div class="section">
    <h2>Scanner Table</h2>
    {table_html}
  </div>

  <div class="section">
    <h2>Top 3 Performance Log — {config['label']}</h2>
    <p>Latest 60 {config['label'].lower()} ideas are shown first.</p>
    {perf_html}
  </div>
</section>
"""


def render_html(scans):
    now_ist = datetime.now(IST)
    generated_time = now_ist.strftime("%d %b %Y, %I:%M:%S %p IST")
    build_id = now_ist.strftime("%Y%m%d%H%M%S")

    swing_block = build_mode_block(scans["swing"])
    intraday_block = build_mode_block(scans["intraday"])
    swing_count = len(scans["swing"]["rows"])
    intraday_count = len(scans["intraday"]["rows"])

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>USA Swing Trading Scanner</title>
<style>
body {{
  margin: 0;
  font-family: Arial, sans-serif;
  background: #0f172a;
  color: #e5e7eb;
}}
.topbar {{
  max-width: 1150px;
  margin: 0 auto;
  padding: 18px 18px 0;
}}
.back-link {{
  display: inline-block;
  color: #38bdf8;
  text-decoration: none;
  font-weight: 700;
  font-size: 14px;
  margin-bottom: 6px;
}}
.back-link:hover {{
  color: #7dd3fc;
}}
.container {{
  max-width: 1150px;
  margin: 0 auto;
  padding: 24px 18px 60px;
}}
h1 {{
  font-size: 42px;
  margin-bottom: 6px;
}}
h2 {{
  margin-top: 0;
}}
.subtitle {{
  color: #94a3b8;
  margin-bottom: 18px;
}}
.toggle-wrap {{
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin: 18px 0 20px;
}}
.toggle-btn {{
  background: #111827;
  border: 1px solid #243041;
  color: #e5e7eb;
  border-radius: 999px;
  padding: 11px 18px;
  font-weight: 800;
  cursor: pointer;
}}
.toggle-btn.active {{
  background: #38bdf8;
  border-color: #38bdf8;
  color: #06111f;
}}
.mode-note {{
  color: #94a3b8;
  font-size: 13px;
}}
.mode-panel {{
  display: none;
}}
.mode-panel.active {{
  display: block;
}}
.mode-heading {{
  background: #111827;
  border: 1px solid #243041;
  border-radius: 18px;
  padding: 18px;
  margin-bottom: 18px;
}}
.mode-heading p {{
  color: #94a3b8;
  margin-bottom: 0;
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}}
.stat, .market, .setup, .section {{
  background: #111827;
  border: 1px solid #243041;
  border-radius: 18px;
  padding: 18px;
}}
.stat span {{
  display: block;
  color: #94a3b8;
  font-size: 13px;
}}
.stat b {{
  font-size: 26px;
}}
.market {{
  margin-bottom: 22px;
}}
.market-grid, .grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}}
.grid {{
  grid-template-columns: repeat(3, 1fr);
}}
.market-grid div, .grid div {{
  background: #0b1220;
  border: 1px solid #243041;
  border-radius: 12px;
  padding: 10px;
  color: #94a3b8;
}}
.market-grid b, .grid b {{
  color: #e5e7eb;
}}
.setup-wrap {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin-bottom: 22px;
}}
.section {{
  margin-top: 18px;
  overflow-x: auto;
}}
.data-table {{
  border-collapse: collapse;
  width: 100%;
  min-width: 1150px;
  font-size: 14px;
}}
.data-table th, .data-table td {{
  border-bottom: 1px solid #243041;
  padding: 10px;
  text-align: left;
  white-space: nowrap;
}}
.data-table th {{
  background: #0b1220;
}}
.disclaimer {{
  color: #94a3b8;
  font-size: 13px;
  line-height: 1.6;
  margin-top: 24px;
}}
@media (max-width: 800px) {{
  .stats, .setup-wrap, .market-grid {{
    grid-template-columns: 1fr;
  }}

  h1 {{
    font-size: 32px;
  }}
}}
</style>
</head>
<body>
<div class="topbar">
  <a class="back-link" href="https://pankajmanchnda.github.io/trading-agents/">
    ← Back to Trading Agents Dashboard
  </a>
</div>
<div class="container">
  <h1>USA Swing Trading Scanner</h1>
  <div class="subtitle">Generated on {generated_time} • Build {build_id} • Swing + Intraday toggle • US large-cap watchlist</div>

  <div class="toggle-wrap">
    <button id="btn-swing" class="toggle-btn" onclick="showMode('swing')">Swing Daily Candle <span>({swing_count})</span></button>
    <button id="btn-intraday" class="toggle-btn active" onclick="showMode('intraday')">Intraday 15-Min Candle <span>({intraday_count})</span></button>
    <span class="mode-note">Toggle changes the displayed scan; both scans are generated during the same build.</span>
  </div>

  <div class="section">
    <h2>Suggested Investment Criteria</h2>
    <p>
      <b>88+</b> Highest Priority ·
      <b>83–87</b> Medium Priority ·
      <b>78–82</b> Low Priority.
      Swing mode uses daily candles. Intraday mode uses 15-minute candles. Entry should only be considered if the trigger level breaks with confirmation.
      Stocks with earnings expected within the next 8 calendar days are excluded.
    </p>
  </div>

  {swing_block}
  {intraday_block}

  <p class="disclaimer">
    Disclaimer: This dashboard is for educational and research purposes only.
    It is not financial advice or a trade recommendation. Yahoo/yfinance intraday data may be delayed or revised.
  </p>
</div>
<script>
function showMode(mode) {{
  const modes = ['swing', 'intraday'];
  modes.forEach(function(item) {{
    const panel = document.getElementById('panel-' + item);
    const button = document.getElementById('btn-' + item);
    if (panel) {{
      panel.classList.toggle('active', item === mode);
    }}
    if (button) {{
      button.classList.toggle('active', item === mode);
    }}
  }});
  const url = new URL(window.location);
  url.searchParams.set('mode', mode);
  window.history.replaceState(null, '', url);
}}
(function initMode() {{
  const params = new URLSearchParams(window.location.search);
  const requested = params.get('mode');
  if (requested === 'swing' || requested === 'intraday') {{
    showMode(requested);
  }} else {{
    showMode('intraday');
  }}
}})();
</script>
</body>
</html>"""

    Path(OUTPUT_HTML).write_text(html, encoding="utf-8")


def main():
    scans = {
        "swing": run_scan("swing"),
        "intraday": run_scan("intraday"),
    }
    render_html(scans)

    print(
        f"Generated {OUTPUT_HTML} with "
        f"{len(scans['swing']['rows'])} swing candidates and "
        f"{len(scans['intraday']['rows'])} intraday candidates. "
        f"Build {datetime.now(IST).strftime('%Y%m%d%H%M%S')}."
    )


if __name__ == "__main__":
    main()
