import math
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# USA Market Scanner
# Swing vs Intraday Toggle - Professional Risk Model + Performance Marker
# =========================

CAPITAL_USD = 100_000
RISK_PER_TRADE = 0.01
MAX_ALLOCATION = 0.20
RR_TARGET = 2.0
TOP_N = 3
HIGHEST_PRIORITY_CAP = 5
EARNINGS_BLACKOUT_DAYS = 8

OUTPUT_HTML = "index.html"
PERFORMANCE_LOG = "performance_log_us.csv"
BENCHMARK = "SPY"
ETF_SYMBOLS = {"SPY", "QQQ", "DIA", "IWM"}
IST = timezone(timedelta(hours=5, minutes=30))

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
UNIVERSE = list(dict.fromkeys(UNIVERSE))

SCAN_MODES = {
    "swing": {
        "label": "Swing",
        "description": "Daily Candle",
        "period": "6mo",
        "interval": "1d",
        "min_rows": 80,
        "bias_change_label": "5-Candle Change",
        "bias_lookback_bars": 6,
        "rs_lookback_bars": 22,
        "structure_lookback": 10,
        "atr_stop_mult": 1.25,
        "atr_structure_buffer": 0.20,
        "min_atr_pct": 1.0,
        "max_atr_pct": 6.0,
        "max_trigger_distance": 1.40,
        "min_vol_ratio": 1.05,
        "min_rs": 1.25,
        "buy_rsi_min": 52,
        "buy_rsi_max": 68,
        "sell_rsi_min": 32,
        "sell_rsi_max": 48,
        "max_extension": 6.0,
    },
    "intraday": {
        "label": "Intraday",
        "description": "15-Min Candle",
        "period": "30d",
        "interval": "15m",
        "min_rows": 80,
        "bias_change_label": "Recent Change",
        "bias_lookback_bars": 22,
        "rs_lookback_bars": 22,
        "structure_lookback": 12,
        "atr_stop_mult": 1.10,
        "atr_structure_buffer": 0.15,
        "min_atr_pct": 0.25,
        "max_atr_pct": 4.0,
        "max_trigger_distance": 0.85,
        "min_vol_ratio": 1.15,
        "min_rs": 0.75,
        "buy_rsi_min": 54,
        "buy_rsi_max": 68,
        "sell_rsi_min": 32,
        "sell_rsi_max": 46,
        "max_extension": 4.5,
    },
}


def now_ist():
    return datetime.now(IST)


def fmt_usd(value):
    try:
        return f"${round(float(value), 0):,.0f}"
    except Exception:
        return "$0"


def fmt_price(value):
    try:
        return round(float(value), 2)
    except Exception:
        return ""


def fmt_distance(value, pct, currency="$"):
    try:
        v = float(value)
        p = float(pct)
        sign = "+" if v >= 0 else "-"
        return f"{sign}{currency}{abs(v):.2f} / {sign}{abs(p):.2f}%"
    except Exception:
        return ""


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
    Stocks with earnings within EARNINGS_BLACKOUT_DAYS are excluded.
    ETFs are not blocked for earnings.
    """
    if ticker in ETF_SYMBOLS:
        return {"skip": False, "status": "ETF - earnings not applicable", "date": ""}

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
    if data is None or data.empty:
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.get_level_values(0):
                return pd.DataFrame()
            df = data[ticker].copy()
        else:
            df = data.copy()

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return pd.DataFrame()

        df = df.dropna(subset=required, how="any")
        df = df[df["Volume"] > 0]
        return df

    except Exception:
        return pd.DataFrame()


def download_data(tickers, mode):
    return yf.download(
        tickers=sorted(set(tickers)),
        period=mode["period"],
        interval=mode["interval"],
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )


def benchmark_bias(bench_df, mode):
    b = bench_df.copy()
    b["EMA20"] = ema(b["Close"], 20)
    b["EMA50"] = ema(b["Close"], 50)

    close = float(b["Close"].iloc[-1])
    ema20 = float(b["EMA20"].iloc[-1])
    ema50 = float(b["EMA50"].iloc[-1])

    lookback = int(mode["bias_lookback_bars"])
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
        message = "US market is mixed/constructive. Scanner will show qualified setups, but entries need stricter confirmation."

    return {
        "bias": bias,
        "message": message,
        "close": round(close, 2),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "recent_change": round(float(recent_change), 2),
        "change_label": mode["bias_change_label"],
    }


def normalise_score(raw_score):
    """
    Setup-quality score, not a win-probability score.
    No live trade should display 100 because no setup is loss-proof.
    """
    raw_score = int(round(raw_score))
    if raw_score >= 98:
        return 94
    if raw_score >= 94:
        return 92
    if raw_score >= 90:
        return 90
    return max(0, min(raw_score, 89))


def score_ticker(df, bench_df, ticker, mode_key):
    mode = SCAN_MODES[mode_key]
    if len(df) < mode["min_rows"] or len(bench_df) < mode["min_rows"]:
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

    rs_lookback = int(mode["rs_lookback_bars"])
    if len(d) < rs_lookback or len(bench_df) < rs_lookback:
        return None

    stock_change = ((d["Close"].iloc[-1] / d["Close"].iloc[-rs_lookback]) - 1) * 100
    bench_change = ((bench_df["Close"].iloc[-1] / bench_df["Close"].iloc[-rs_lookback]) - 1) * 100
    rs_vs_spy = float(stock_change - bench_change)
    if not np.isfinite(rs_vs_spy):
        return None

    ema20_prev = float(d["EMA20"].iloc[-6]) if len(d) >= 26 and np.isfinite(d["EMA20"].iloc[-6]) else ema20
    buy_trend = close > ema20 > ema50 and ema20 > ema20_prev
    sell_trend = close < ema20 < ema50 and ema20 < ema20_prev

    signal = None
    score = 50
    notes = []

    if buy_trend and mode["buy_rsi_min"] <= rsi14 <= mode["buy_rsi_max"] and rs_vs_spy >= mode["min_rs"]:
        signal = "BUY"
        score += 16
        notes.append("Confirmed uptrend")
    elif sell_trend and mode["sell_rsi_min"] <= rsi14 <= mode["sell_rsi_max"] and rs_vs_spy <= -mode["min_rs"]:
        signal = "SELL"
        score += 16
        notes.append("Confirmed downtrend")
    else:
        return None

    recent = d.tail(int(mode["structure_lookback"]))
    last_high = float(d["High"].iloc[-1])
    last_low = float(d["Low"].iloc[-1])
    recent_low = float(recent["Low"].min())
    recent_high = float(recent["High"].max())

    if signal == "BUY":
        extension_pct = ((close / ema20) - 1) * 100 if ema20 > 0 else np.nan
        if not np.isfinite(extension_pct) or extension_pct > mode["max_extension"]:
            return None

        entry = max(close, last_high * 1.002)
        atr_stop = entry - float(mode["atr_stop_mult"]) * atr14
        structure_stop = recent_low - float(mode["atr_structure_buffer"]) * atr14
        stop = min(atr_stop, structure_stop)
        entry_rule = "Enter only above trigger; prefer candle close confirmation"
    else:
        extension_pct = ((ema20 / close) - 1) * 100 if close > 0 else np.nan
        if not np.isfinite(extension_pct) or extension_pct > mode["max_extension"]:
            return None

        entry = min(close, last_low * 0.998)
        atr_stop = entry + float(mode["atr_stop_mult"]) * atr14
        structure_stop = recent_high + float(mode["atr_structure_buffer"]) * atr14
        stop = max(atr_stop, structure_stop)
        entry_rule = "Enter only below trigger; prefer candle close confirmation"

    trigger_distance_pct = abs(entry - close) / close * 100 if close > 0 else np.nan
    if not np.isfinite(trigger_distance_pct):
        return None

    if trigger_distance_pct > mode["max_trigger_distance"]:
        return None
    if vol_ratio < mode["min_vol_ratio"]:
        return None
    if not (mode["min_atr_pct"] <= atr_pct <= mode["max_atr_pct"]):
        return None

    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return None

    # Avoid setups where structure-aware stop is impractically wide.
    risk_pct = (risk_per_share / entry) * 100 if entry > 0 else np.nan
    if not np.isfinite(risk_pct):
        return None
    if mode_key == "swing" and risk_pct > 8.0:
        return None
    if mode_key == "intraday" and risk_pct > 4.5:
        return None

    if signal == "BUY":
        target = entry + RR_TARGET * risk_per_share
    else:
        target = entry - RR_TARGET * risk_per_share

    if signal == "BUY":
        if rs_vs_spy >= mode["min_rs"] * 2:
            score += 12
            notes.append("Strong relative strength")
        else:
            score += 7
            notes.append("Positive relative strength")

        if 54 <= rsi14 <= 66:
            score += 7
            notes.append("Healthy RSI")
        else:
            score += 4
            notes.append("Acceptable RSI")
    else:
        if rs_vs_spy <= -mode["min_rs"] * 2:
            score += 12
            notes.append("Weak relative strength")
        else:
            score += 7
            notes.append("Negative relative strength")

        if 34 <= rsi14 <= 46:
            score += 7
            notes.append("Healthy RSI")
        else:
            score += 4
            notes.append("Acceptable RSI")

    if vol_ratio >= 1.35:
        score += 8
        notes.append("Institutional volume")
    elif vol_ratio >= 1.10:
        score += 5
        notes.append("Good volume")
    else:
        score += 2
        notes.append("Acceptable volume")

    if mode_key == "swing":
        if 1.2 <= atr_pct <= 4.8:
            score += 6
            notes.append("Clean swing volatility")
        else:
            score += 3
            notes.append("Acceptable volatility")
    else:
        if 0.35 <= atr_pct <= 3.2:
            score += 6
            notes.append("Clean intraday volatility")
        else:
            score += 3
            notes.append("Acceptable volatility")

    if trigger_distance_pct <= mode["max_trigger_distance"] * 0.55:
        score += 5
        notes.append("Close to trigger")
    else:
        score += 2
        notes.append("Trigger needs confirmation")

    if extension_pct <= mode["max_extension"] * 0.55:
        score += 5
        notes.append("Not extended")
    else:
        score += 2
        notes.append("Slightly extended")

    score = normalise_score(score)

    if score < 78:
        return None

    max_risk_dollars = CAPITAL_USD * RISK_PER_TRADE
    qty_by_risk = math.floor(max_risk_dollars / risk_per_share)
    qty_by_allocation = math.floor((CAPITAL_USD * MAX_ALLOCATION) / entry) if entry > 0 else 0
    qty = max(0, min(qty_by_risk, qty_by_allocation))
    trade_value = qty * entry
    if qty <= 0 or trade_value <= 0:
        return None

    return {
        "Mode": mode["label"],
        "Priority": "Qualified Candidate",
        "Stock": ticker,
        "Signal": signal,
        "Conviction": int(score),
        "Grade": "A" if score >= 90 else "B+" if score >= 84 else "B",
        "Close": round(close, 2),
        "Entry": round(entry, 2),
        "Stop": round(stop, 2),
        "Target": round(target, 2),
        "RR": RR_TARGET,
        "RSI": round(rsi14, 1),
        "ATR%": f"{round(atr_pct, 2)}%",
        "Vol Ratio": round(vol_ratio, 2),
        "RS vs SPY": round(rs_vs_spy, 2),
        "Trigger Distance": f"{round(trigger_distance_pct, 2)}%",
        "Extension": f"{round(extension_pct, 2)}%",
        "Risk %": f"{round(risk_pct, 2)}%",
        "Earnings": "Clear",
        "Earnings Date": "",
        "Qty": qty,
        "Trade Value": fmt_usd(trade_value),
        "Entry Rule": entry_rule,
        "Notes": " · ".join(notes),
    }


def calibrate_priorities(rows, highest_cap=HIGHEST_PRIORITY_CAP):
    calibrated = []
    for rank, row in enumerate(sorted(rows, key=lambda x: x["Conviction"], reverse=True), start=1):
        score = int(row["Conviction"])
        if rank <= highest_cap and score >= 90:
            row["Priority"] = "Highest Priority"
            row["Grade"] = "A"
        elif score >= 84:
            row["Priority"] = "Medium Priority"
            row["Grade"] = "B+"
        elif score >= 78:
            row["Priority"] = "Low Priority"
            row["Grade"] = "B"
        else:
            continue
        calibrated.append(row)
    return calibrated


def calculate_distances(current_price, signal, stop, target):
    current_price = float(current_price)
    stop = float(stop)
    target = float(target)

    if signal == "BUY":
        stop_value = current_price - stop
        stop_pct = (stop_value / current_price) * 100 if current_price else np.nan
        target_value = target - current_price
        target_pct = (target_value / current_price) * 100 if current_price else np.nan
    else:
        stop_value = stop - current_price
        stop_pct = (stop_value / current_price) * 100 if current_price else np.nan
        target_value = current_price - target
        target_pct = (target_value / current_price) * 100 if current_price else np.nan

    return (
        fmt_distance(stop_value, stop_pct),
        fmt_distance(target_value, target_pct),
    )


def assess_trade_path(df, row):
    """
    Determine whether the entry triggered, then whether stop or target was touched first.
    If entry/stop/target occur in the same candle, mark ambiguous instead of assuming a loss or win.
    """
    if df is None or df.empty:
        return {
            "current_price": "",
            "status": row.get("Status", "Pending Trigger"),
            "first_hit": row.get("First Hit", ""),
            "hit_date": row.get("Hit Date", ""),
            "result": row.get("Result", ""),
            "r": row.get("R", ""),
            "days": row.get("Days", ""),
            "stop_away": "",
            "target_away": "",
        }

    signal = str(row.get("Signal", "")).upper()
    entry = pd.to_numeric(row.get("Entry", np.nan), errors="coerce")
    stop = pd.to_numeric(row.get("Stop", np.nan), errors="coerce")
    target = pd.to_numeric(row.get("Target", np.nan), errors="coerce")

    if not all(np.isfinite(x) for x in [entry, stop, target]) or signal not in {"BUY", "SELL"}:
        return {
            "current_price": "",
            "status": row.get("Status", "Pending Trigger"),
            "first_hit": row.get("First Hit", ""),
            "hit_date": row.get("Hit Date", ""),
            "result": row.get("Result", ""),
            "r": row.get("R", ""),
            "days": row.get("Days", ""),
            "stop_away": "",
            "target_away": "",
        }

    current_price = float(df["Close"].iloc[-1])
    stop_away, target_away = calculate_distances(current_price, signal, stop, target)

    date_str = str(row.get("Date", ""))
    try:
        start_date = pd.to_datetime(date_str).date()
    except Exception:
        start_date = None

    path = df.copy()
    if start_date is not None:
        idx_dates = pd.to_datetime(path.index).date
        path = path[idx_dates >= start_date]

    if path.empty:
        return {
            "current_price": round(current_price, 2),
            "status": row.get("Status", "Pending Trigger"),
            "first_hit": row.get("First Hit", ""),
            "hit_date": row.get("Hit Date", ""),
            "result": row.get("Result", ""),
            "r": row.get("R", ""),
            "days": row.get("Days", ""),
            "stop_away": stop_away,
            "target_away": target_away,
        }

    entry_triggered = False
    entry_bar_index = None

    for i, (_, candle) in enumerate(path.iterrows()):
        high = float(candle["High"])
        low = float(candle["Low"])
        if signal == "BUY" and high >= entry:
            entry_triggered = True
            entry_bar_index = i
            break
        if signal == "SELL" and low <= entry:
            entry_triggered = True
            entry_bar_index = i
            break

    if not entry_triggered:
        days = (now_ist().date() - start_date).days if start_date else ""
        return {
            "current_price": round(current_price, 2),
            "status": "Pending Trigger",
            "first_hit": "",
            "hit_date": "",
            "result": "",
            "r": "",
            "days": days,
            "stop_away": stop_away,
            "target_away": target_away,
        }

    triggered_path = path.iloc[entry_bar_index:]
    for j, (idx, candle) in enumerate(triggered_path.iterrows()):
        high = float(candle["High"])
        low = float(candle["Low"])
        hit_date = pd.to_datetime(idx).date().isoformat()

        if signal == "BUY":
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            stop_hit = high >= stop
            target_hit = low <= target

        if j == 0 and (stop_hit or target_hit):
            return {
                "current_price": round(current_price, 2),
                "status": "Ambiguous Entry Candle",
                "first_hit": "Ambiguous",
                "hit_date": hit_date,
                "result": "Review",
                "r": "",
                "days": (pd.to_datetime(idx).date() - start_date).days if start_date else "",
                "stop_away": stop_away,
                "target_away": target_away,
            }

        if stop_hit and target_hit:
            return {
                "current_price": round(current_price, 2),
                "status": "Ambiguous Same Candle",
                "first_hit": "Ambiguous",
                "hit_date": hit_date,
                "result": "Review",
                "r": "",
                "days": (pd.to_datetime(idx).date() - start_date).days if start_date else "",
                "stop_away": stop_away,
                "target_away": target_away,
            }

        if target_hit:
            return {
                "current_price": round(current_price, 2),
                "status": "Target Hit First",
                "first_hit": "Target",
                "hit_date": hit_date,
                "result": "Win",
                "r": RR_TARGET,
                "days": (pd.to_datetime(idx).date() - start_date).days if start_date else "",
                "stop_away": stop_away,
                "target_away": target_away,
            }

        if stop_hit:
            return {
                "current_price": round(current_price, 2),
                "status": "Stop Hit First",
                "first_hit": "Stop",
                "hit_date": hit_date,
                "result": "Loss",
                "r": -1,
                "days": (pd.to_datetime(idx).date() - start_date).days if start_date else "",
                "stop_away": stop_away,
                "target_away": target_away,
            }

    days = (now_ist().date() - start_date).days if start_date else ""
    return {
        "current_price": round(current_price, 2),
        "status": "Active After Trigger",
        "first_hit": "",
        "hit_date": "",
        "result": "",
        "r": "",
        "days": days,
        "stop_away": stop_away,
        "target_away": target_away,
    }


def refresh_performance_status(log, data_by_mode):
    if log.empty:
        return log

    # Use object dtype for assignment-safe mixed string/numeric display values.
    for col in log.columns:
        log[col] = log[col].astype("object")

    for idx, row in log.iterrows():
        mode_label = str(row.get("Mode", ""))
        stock = str(row.get("Stock", ""))
        mode_key = "swing" if mode_label.lower() == "swing" else "intraday" if mode_label.lower() == "intraday" else None
        if not mode_key or not stock:
            continue

        data = data_by_mode.get(mode_key)
        df = get_single_df(data, stock)
        if df.empty:
            continue

        result = assess_trade_path(df, row)
        log.at[idx, "Current Price"] = result["current_price"]
        log.at[idx, "Status"] = result["status"]
        log.at[idx, "First Hit"] = result["first_hit"]
        log.at[idx, "Hit Date"] = result["hit_date"]
        log.at[idx, "Result"] = result["result"]
        log.at[idx, "R"] = result["r"]
        log.at[idx, "Days"] = result["days"]
        log.at[idx, "Stop Away"] = result["stop_away"]
        log.at[idx, "Target Away"] = result["target_away"]

    return log


def update_performance_log(all_rows_by_mode, data_by_mode):
    log_path = Path(PERFORMANCE_LOG)
    today = now_ist().date().isoformat()

    columns = [
        "Date", "Mode", "Rank", "Stock", "Signal", "Conviction", "Entry", "Stop",
        "Target", "RR", "Current Price", "Status", "First Hit", "Hit Date",
        "Stop Away", "Target Away", "Result", "R", "Days", "Notes"
    ]

    if log_path.exists():
        log = pd.read_csv(log_path)
        if "Mode" not in log.columns:
            log["Mode"] = "Legacy"
        for col in columns:
            if col not in log.columns:
                log[col] = ""
        log = log[columns]
        log["Conviction"] = pd.to_numeric(log["Conviction"], errors="coerce").clip(upper=94)
    else:
        log = pd.DataFrame(columns=columns)

    for col in columns:
        log[col] = log[col].astype("object")

    existing = set(zip(log.get("Date", []), log.get("Mode", []), log.get("Rank", []), log.get("Stock", [])))

    new_rows = []
    for mode_key, rows in all_rows_by_mode.items():
        mode_label = SCAN_MODES[mode_key]["label"]
        for rank, row in enumerate(rows[:TOP_N], start=1):
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
                    "Current Price": row["Close"],
                    "Status": "Pending Trigger",
                    "First Hit": "",
                    "Hit Date": "",
                    "Stop Away": "",
                    "Target Away": "",
                    "Result": "",
                    "R": "",
                    "Days": 0,
                    "Notes": f"Top 3 {mode_label} scanner pick",
                })

    if new_rows:
        log = pd.concat([pd.DataFrame(new_rows), log], ignore_index=True)
        for col in columns:
            log[col] = log[col].astype("object")

    log = refresh_performance_status(log, data_by_mode)
    log = log[columns]
    log.to_csv(log_path, index=False)
    return log.head(120)


def run_scan(mode_key):
    mode = SCAN_MODES[mode_key]
    tickers = sorted(set(UNIVERSE + [BENCHMARK]))

    print(f"Downloading {mode['label']} data: period={mode['period']}, interval={mode['interval']}...")
    data = download_data(tickers, mode)

    bench_df = get_single_df(data, BENCHMARK)
    if bench_df.empty:
        raise RuntimeError(f"{BENCHMARK} benchmark data could not be downloaded for {mode['label']} mode.")

    market = benchmark_bias(bench_df, mode)
    rows = []

    for ticker in sorted(set(UNIVERSE)):
        if ticker == BENCHMARK:
            continue

        earnings = get_earnings_risk(ticker)
        if earnings["skip"]:
            continue

        df = get_single_df(data, ticker)
        if df.empty:
            continue

        result = score_ticker(df, bench_df, ticker, mode_key)
        if not result:
            continue

        if market["bias"] == "Bullish" and result["Signal"] != "BUY":
            continue
        if market["bias"] == "Bearish" and result["Signal"] != "SELL":
            continue
        if market["bias"] == "Mixed" and result["Conviction"] < 84:
            continue

        result["Earnings"] = earnings["status"]
        result["Earnings Date"] = earnings["date"]
        rows.append(result)

    rows = calibrate_priorities(rows, highest_cap=HIGHEST_PRIORITY_CAP)
    rows = rows[:20]
    print(f"{mode['label']}: generated {len(rows)} candidates.")
    return rows, market, data


def df_to_html(rows):
    if not rows:
        return "<p>No qualifying setups in this mode right now.</p>"
    return pd.DataFrame(rows).to_html(index=False, classes="data-table", escape=False)


def perf_to_html(perf_log, mode_label):
    if perf_log.empty:
        return "<p>No records yet.</p>"
    df = perf_log[perf_log["Mode"] == mode_label].head(60)
    if df.empty:
        return "<p>No records yet for this mode.</p>"
    return df.to_html(index=False, classes="data-table", escape=False)


def render_cards(rows):
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
    return cards or "<p>No fresh qualifying setups in this mode right now. Historical ideas below are from earlier runs and may no longer be valid.</p>"


def render_mode_block(mode_key, rows, market, perf_log, active=False):
    mode = SCAN_MODES[mode_key]
    mode_label = mode["label"]
    total = len(rows)
    high = sum(r["Priority"] == "Highest Priority" for r in rows)
    medium = sum(r["Priority"] == "Medium Priority" for r in rows)
    low = sum(r["Priority"] == "Low Priority" for r in rows)
    best = max([r["Conviction"] for r in rows], default=0)
    active_class = "active" if active else ""

    return f"""
    <section id="{mode_key}" class="mode-panel {active_class}">
      <div class="mode-heading">
        <h2>{mode_label} Scanner</h2>
        <p>{mode['description']} · Period {mode['period']} · Interval {mode['interval']}</p>
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
      <div class="setup-wrap">{render_cards(rows)}</div>

      <div class="section">
        <h2>Suggested Investment Criteria</h2>
        <p>
          <b>90–94</b> Highest Priority candidate range, but only the top 5 ranked ideas per mode can receive that label ·
          <b>84–89</b> Medium Priority ·
          <b>78–83</b> Low Priority.
          Score is setup quality, not probability of profit. No trade is assumed loss-proof, so the model intentionally avoids 100 scores.
          Expert filter requires trend alignment, SPY relative strength, volume confirmation, controlled volatility, and a nearby trigger.
          Entry should only be considered if the trigger level breaks with confirmation. Stocks with earnings expected within the next 8 calendar days are excluded.
        </p>
      </div>

      <div class="section">
        <h2>Scanner Table</h2>
        {df_to_html(rows)}
      </div>

      <div class="section">
        <h2>{mode_label} Top 3 Performance Log</h2>
        <p>Latest 60 ideas for this mode are shown first. Current Price, First Hit, Hit Date, Stop Away, and Target Away update on each scanner run to help track whether target or stop-loss was reached first.</p>
        {perf_to_html(perf_log, mode_label)}
      </div>
    </section>
    """


def render_html(all_rows_by_mode, markets_by_mode, perf_log):
    generated_at = now_ist()
    generated_time = generated_at.strftime("%d %b %Y, %I:%M:%S %p IST")
    build_id = generated_at.strftime("%Y%m%d%H%M%S")

    swing_block = render_mode_block(
        "swing", all_rows_by_mode["swing"], markets_by_mode["swing"], perf_log, active=True
    )
    intraday_block = render_mode_block(
        "intraday", all_rows_by_mode["intraday"], markets_by_mode["intraday"], perf_log, active=False
    )

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
  color: #38bdf8;
  text-decoration: none;
  font-weight: 700;
  font-size: 14px;
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
  font-size: 32px;
  margin: 0 0 6px;
}}

h2 {{
  font-size: 20px;
}}

.subtitle {{
  color: #94a3b8;
  margin-bottom: 14px;
  line-height: 1.5;
}}

.refresh-panel {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 14px 0 18px 0;
  color: #94a3b8;
  font-size: 13px;
  flex-wrap: wrap;
}}

.refresh-panel button {{
  background: #111827;
  border: 1px solid #243041;
  color: #e5e7eb;
  border-radius: 999px;
  padding: 8px 14px;
  cursor: pointer;
  font-weight: 700;
}}

.refresh-panel button.active {{
  background: #38bdf8;
  color: #020617;
  border-color: #38bdf8;
}}

.refresh-note {{
  color: #64748b;
}}

.toggle-wrap {{
  display: flex;
  gap: 10px;
  margin: 18px 0 24px;
  flex-wrap: wrap;
}}

.toggle-btn {{
  background: #111827;
  color: #e5e7eb;
  border: 1px solid #243041;
  border-radius: 999px;
  padding: 11px 18px;
  cursor: pointer;
  font-weight: 700;
}}

.toggle-btn.active {{
  background: #38bdf8;
  color: #07111f;
  border-color: #38bdf8;
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
  font-size: 24px;
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
  min-width: 1350px;
  font-size: 13px;
}}

.data-table th, .data-table td {{
  border-bottom: 1px solid #243041;
  padding: 9px;
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
  h1 {{
    font-size: 26px;
  }}

  .stats,
  .setup-wrap,
  .market-grid {{
    grid-template-columns: 1fr;
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
  <div class="subtitle">
    Generated on {generated_time} · Build {build_id} · SPY benchmark · Swing vs Intraday scanner
  </div>

  <div class="refresh-panel">
    <button id="autoRefreshBtn" type="button">Auto-refresh OFF</button>
    <span id="refreshCountdown">Manual refresh mode</span>
    <span class="refresh-note">Reloads this page every 5 minutes when enabled.</span>
  </div>

  <div class="toggle-wrap">
    <button class="toggle-btn active" onclick="showMode('swing', this)">Swing Daily Candle</button>
    <button class="toggle-btn" onclick="showMode('intraday', this)">Intraday 15-Min Candle</button>
  </div>

  {swing_block}
  {intraday_block}

  <p class="disclaimer">
    Disclaimer: This dashboard is for educational and research purposes only.
    Scores are setup-quality rankings, not win probabilities. No market setup is guaranteed or loss-proof.
    This is not financial advice or a trade recommendation.
    Intraday prices from Yahoo Finance may be delayed depending on exchange/feed availability.
  </p>
</div>

<script>
function showMode(mode, button) {{
  document.querySelectorAll('.mode-panel').forEach(function(panel) {{
    panel.classList.remove('active');
  }});

  document.querySelectorAll('.toggle-btn').forEach(function(btn) {{
    btn.classList.remove('active');
  }});

  document.getElementById(mode).classList.add('active');
  button.classList.add('active');

  const url = new URL(window.location.href);
  url.searchParams.set('mode', mode);
  window.history.replaceState(null, '', url.toString());
}}

(function initMode() {{
  const params = new URLSearchParams(window.location.search);
  const requested = params.get('mode');
  if (requested === 'swing' || requested === 'intraday') {{
    const btn = Array.from(document.querySelectorAll('.toggle-btn')).find(function(button) {{
      return button.textContent.toLowerCase().includes(requested);
    }});
    if (btn) showMode(requested, btn);
  }}
}})();

(function () {{
  const REFRESH_SECONDS = 300;
  const STORAGE_KEY = "scanner_us_auto_refresh_enabled";

  const btn = document.getElementById("autoRefreshBtn");
  const countdown = document.getElementById("refreshCountdown");

  if (!btn || !countdown) return;

  let enabled = localStorage.getItem(STORAGE_KEY) === "true";
  let remaining = REFRESH_SECONDS;
  let timer = null;

  function formatTime(seconds) {{
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${{mins}}:${{String(secs).padStart(2, "0")}}`;
  }}

  function updateUi() {{
    if (enabled) {{
      btn.textContent = "Auto-refresh ON";
      btn.classList.add("active");
      countdown.textContent = `Next refresh in ${{formatTime(remaining)}}`;
    }} else {{
      btn.textContent = "Auto-refresh OFF";
      btn.classList.remove("active");
      countdown.textContent = "Manual refresh mode";
    }}
  }}

  function startTimer() {{
    if (timer) clearInterval(timer);

    timer = setInterval(function () {{
      if (!enabled) return;

      remaining -= 1;
      if (remaining <= 0) {{
        const url = new URL(window.location.href);
        url.searchParams.set("v", Date.now().toString());
        window.location.href = url.toString();
        return;
      }}

      updateUi();
    }}, 1000);
  }}

  btn.addEventListener("click", function () {{
    enabled = !enabled;
    remaining = REFRESH_SECONDS;
    localStorage.setItem(STORAGE_KEY, enabled ? "true" : "false");
    updateUi();
    startTimer();
  }});

  updateUi();
  startTimer();
}})();
</script>
</body>
</html>"""

    Path(OUTPUT_HTML).write_text(html, encoding="utf-8")


def main():
    all_rows_by_mode = {}
    markets_by_mode = {}
    data_by_mode = {}

    for mode_key in SCAN_MODES:
        rows, market, data = run_scan(mode_key)
        all_rows_by_mode[mode_key] = rows
        markets_by_mode[mode_key] = market
        data_by_mode[mode_key] = data

    perf_log = update_performance_log(all_rows_by_mode, data_by_mode)
    render_html(all_rows_by_mode, markets_by_mode, perf_log)

    print(f"Generated {OUTPUT_HTML}. Build {now_ist().strftime('%Y%m%d%H%M%S')}.")


if __name__ == "__main__":
    main()
