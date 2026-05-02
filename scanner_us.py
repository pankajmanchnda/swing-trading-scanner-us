def score_ticker(df, bench_df, ticker):
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

    stock_21 = ((d["Close"].iloc[-1] / d["Close"].iloc[-22]) - 1) * 100
    bench_21 = ((bench_df["Close"].iloc[-1] / bench_df["Close"].iloc[-22]) - 1) * 100
    rs_vs_bench = float(stock_21 - bench_21)

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

    # Stricter US-market quality filters
    if trigger_distance_pct > 1.25:
        return None

    if vol_ratio < 1.05:
        return None

    if signal == "BUY" and rs_vs_bench < 1.5:
        return None

    if signal == "SELL" and rs_vs_bench > -1.5:
        return None

    if atr_pct > 6:
        return None

    # Stricter scoring thresholds
    if score >= 90:
        priority = "Highest Priority"
        grade = "A"
    elif score >= 86:
        priority = "Medium Priority"
        grade = "B+"
    elif score >= 82:
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
        "Qty": qty,
        "Trade Value": f"${round(trade_value, 0):,.0f}",
        "Entry Rule": entry_rule,
        "Notes": " · ".join(notes),
    }
