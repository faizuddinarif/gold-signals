"""
Confluence signal bot -> Telegram push notifications (free).
Same logic as the TradingView indicator: 7-factor score, A/B grade,
ATR-based SL + TP1-3, break-even/trail management.

Env vars:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID   (required to send; otherwise prints only)
  SYMBOL    default GC=F  (Yahoo gold futures; try XAUUSD=X for spot)
  INTERVAL  5m | 15m | 30m | 1h | 1d   (default 5m)
  MIN_SCORE default 6.0   GRADE_A default 8.0   GRADE_FILTER all | A
Run once:  python bot.py        Run forever: python bot.py --loop
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import requests

SYMBOL = os.getenv("SYMBOL", "GC=F")
INTERVAL = os.getenv("INTERVAL", "5m")
MIN_SCORE = float(os.getenv("MIN_SCORE", "6.0"))
GRADE_A = float(os.getenv("GRADE_A", "8.0"))
GRADE_FILTER = os.getenv("GRADE_FILTER", "all").lower()
ADX_MIN = 20.0
STATE_FILE = os.getenv("STATE_FILE", "signals.json")

PRESETS = {
    "Scalping": dict(fast=8, slow=21, sl=1.2, tp=(1.0, 2.0, 3.0)),
    "Default": dict(fast=13, slow=34, sl=1.5, tp=(1.5, 3.0, 5.0)),
    "Swing": dict(fast=21, slow=55, sl=2.0, tp=(2.0, 4.0, 6.0)),
}
PRESET_BY_INTERVAL = {"5m": "Scalping", "15m": "Scalping", "30m": "Default", "1h": "Default", "1d": "Swing"}
FETCH = {  # interval: (period, htf_interval, htf_period)
    "5m": ("5d", "1h", "60d"), "15m": ("30d", "1h", "60d"),
    "30m": ("30d", "1d", "1y"), "1h": ("60d", "1d", "1y"), "1d": ("2y", "1wk", "5y"),
}

# ───────────── indicators ─────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rma(s, n): return s.ewm(alpha=1 / n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    up, dn = rma(d.clip(lower=0), n), rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def true_range(df):
    pc = df["Close"].shift(1)
    return pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)

def adx(df, n=14):
    up, dn = df["High"].diff(), -df["Low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = rma(true_range(df), n)
    pdi, mdi = 100 * rma(plus, n) / atr, 100 * rma(minus, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx.fillna(0), n)

def supertrend_bull(df, factor=3.0, n=10):
    atr = rma(true_range(df), n)
    hl2 = (df["High"] + df["Low"]) / 2
    up, lo, c = (hl2 + factor * atr).values, (hl2 - factor * atr).values, df["Close"].values
    bull = np.ones(len(df), dtype=bool)
    fu, fl = up[0], lo[0]
    for i in range(1, len(df)):
        fu_p, fl_p = fu, fl
        fu = up[i] if (up[i] < fu_p or c[i - 1] > fu_p) else fu_p
        fl = lo[i] if (lo[i] > fl_p or c[i - 1] < fl_p) else fl_p
        if bull[i - 1] and c[i] < fl_p: bull[i] = False
        elif (not bull[i - 1]) and c[i] > fu_p: bull[i] = True
        else: bull[i] = bull[i - 1]
    return pd.Series(bull, index=df.index)

def build(df, preset, htf_bull, htf_bear):
    p = PRESETS[preset]
    o = pd.DataFrame(index=df.index)
    o["emaF"], o["emaS"] = ema(df["Close"], p["fast"]), ema(df["Close"], p["slow"])
    o["rsi"], o["adx"] = rsi(df["Close"]), adx(df)
    o["atr"] = rma(true_range(df), 14)
    o["atrRatio"] = o["atr"] / o["atr"].rolling(50).mean()
    macd = ema(df["Close"], 12) - ema(df["Close"], 26)
    h = macd - ema(macd, 9)
    st_bull = supertrend_bull(df)

    o["bullTrend"] = o["emaF"] > o["emaS"]
    cA = o["adx"] > ADX_MIN
    cV = o["atrRatio"].between(0.6, 1.8)
    f = {  # factor -> (long, short)
        "Trend": (o["bullTrend"], ~o["bullTrend"]),
        "HTF": (pd.Series(htf_bull, index=df.index), pd.Series(htf_bear, index=df.index)),
        "Supertrend": (st_bull, ~st_bull),
        "RSI": ((o["rsi"] > 50) & (o["rsi"] < 75), (o["rsi"] < 50) & (o["rsi"] > 25)),
        "ADX": (cA, cA),
        "Volatility": (cV, cV),
        "MACD": ((h > 0) & (h > h.shift(1)), (h < 0) & (h < h.shift(1))),
    }
    w = {"Trend": 2, "HTF": 2, "Supertrend": 1, "RSI": 1, "ADX": 1.5, "Volatility": 1, "MACD": 1.5}
    o["longScore"] = sum(f[k][0].astype(float) * w[k] for k in w)
    o["shortScore"] = sum(f[k][1].astype(float) * w[k] for k in w)
    o["score"] = np.where(o["bullTrend"], o["longScore"], o["shortScore"])
    ok = (o["score"] >= GRADE_A) if GRADE_FILTER.startswith("a") else True
    o["longSig"] = o["bullTrend"] & (o["longScore"] >= MIN_SCORE) & (o["longScore"].shift(1) < MIN_SCORE) & ok
    o["shortSig"] = (~o["bullTrend"]) & (o["shortScore"] >= MIN_SCORE) & (o["shortScore"].shift(1) < MIN_SCORE) & ok
    return o, f, w

# ───────────── messages ─────────────
def px(v): return f"{v:,.2f}"

def factor_names(o, i):
    return {"Trend": "Trend (EMA)", "HTF": "Higher-TF bias", "Supertrend": "Supertrend",
            "RSI": f"RSI {o['rsi'].iloc[i]:.1f}", "ADX": f"ADX {o['adx'].iloc[i]:.1f}",
            "Volatility": "Volatility", "MACD": "MACD momentum"}

def card(t, o, row_i, f, w, label):
    d = t["dir"]
    side = "🟢 BUY" if d == 1 else "🔴 SELL"
    score = t["score"]
    bars = "█" * round(score) + "░" * (10 - round(score))
    r = t["atrRatio"]
    risk = "LOW RISK" if r < 0.8 else "MODERATE RISK" if r <= 1.3 else "HIGH RISK"
    lines = [
        f"<b>{side}  {label}</b>  ·  Grade {t['grade']}",
        f"Confidence: {bars} {score * 10:.0f}%  ({score:g}/10)",
        f"⚠️ {risk}", "",
        f"<b>Entry:</b> {px(t['entry'])}",
        f"<b>Stop loss:</b> {px(t['sl'])}  (risk {px(abs(t['entry'] - t['sl']))})",
        f"<b>TP1:</b> {px(t['tp1'])}",
        f"<b>TP2:</b> {px(t['tp2'])}",
        f"<b>TP3:</b> {px(t['tp3'])}",
        f"R:R to TP3 = 1 : {t['rr']:.1f}",
        "Plan: SL to BE at TP1, trail to TP1 at TP2", "", "<b>Factors</b>"]
    names = factor_names(o, row_i)
    for k in w:
        ok = bool(f[k][0 if d == 1 else 1].iloc[row_i])
        lines.append(f"{'✅' if ok else '❌'} {names[k]} ({w[k]:g})")
    lines.append(f"\n🕒 {t['time']}")
    return "\n".join(lines)

# ───────────── core: process new closed bars ─────────────
def process(df, htf_bull, htf_bear, state, label=None):
    preset = PRESET_BY_INTERVAL.get(INTERVAL, "Default")
    p = PRESETS[preset]
    label = label or f"XAUUSD {INTERVAL}"
    o, f, w = build(df, preset, htf_bull, htf_bear)
    msgs = []

    if not state:
        state = {"last_ts": df.index[-1].isoformat(), "trade": None}
        msgs.append(f"🤖 Signal bot started: {label} ({preset}). Watching for setups.")
    state.setdefault("history", [])
    state.setdefault("counter", 0)

    last = pd.Timestamp(state["last_ts"])
    t = state.get("trade")
    for i in range(len(df)):
        ts = df.index[i]
        if ts <= last: continue
        hi, lo, cl = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i]
        evt, closed = [], False
        if t:
            d = t["dir"]
            sl_hit = lo <= t["sl"] if d == 1 else hi >= t["sl"]
            hit = (lambda lv: hi >= lv) if d == 1 else (lambda lv: lo <= lv)
            if sl_hit:
                txt = ["Stop loss hit ❌", "Stopped at break-even ➖", "Trail stop hit at TP1 (profit locked ✅)"][t["stage"]]
                res, exitp = ["SL", "BE", "TP1 locked"][t["stage"]], t["sl"]
                evt.append(f"{txt}\nPrice: {px(t['sl'])}")
                closed = True
            else:
                if t["stage"] < 1 and hit(t["tp1"]):
                    t["stage"], t["sl"] = 1, t["entry"]
                    evt.append(f"🎯 TP1 hit {px(t['tp1'])} ✅\nSL moved to break-even: {px(t['entry'])}")
                if t["stage"] < 2 and hit(t["tp2"]):
                    t["stage"], t["sl"] = 2, t["tp1"]
                    evt.append(f"🎯 TP2 hit {px(t['tp2'])} ✅\nSL trailed to TP1: {px(t['tp1'])}")
                if hit(t["tp3"]):
                    evt.append(f"🏆 TP3 hit {px(t['tp3'])} ✅ Trade complete")
                    res, exitp = "TP3", t["tp3"]
                    closed = True
            if closed:
                risk0 = abs(t["entry"] - t["sl0"])
                state["history"].append({**t, "closed": ts.isoformat(), "result": res,
                                         "r": round(d * (exitp - t["entry"]) / risk0, 2)})
                state["history"] = state["history"][-30:]
                t = None
        if evt:
            msgs.append(f"📣 <b>TRADE UPDATE | {label}</b>\n" + "\n".join(evt))
        if t is None and not closed and (o["longSig"].iloc[i] or o["shortSig"].iloc[i]):
            d = 1 if o["longSig"].iloc[i] else -1
            a, sc = o["atr"].iloc[i], float(o["score"].iloc[i])
            t = dict(dir=d, stage=0, entry=cl, sl=cl - d * a * p["sl"],
                     tp1=cl + d * a * p["tp"][0], tp2=cl + d * a * p["tp"][1], tp3=cl + d * a * p["tp"][2],
                     score=sc, grade="A" if sc >= GRADE_A else "B", atrRatio=float(o["atrRatio"].iloc[i]),
                     rr=p["tp"][2] / p["sl"], time=ts.strftime("%Y-%m-%d %H:%M %Z"))
            state["counter"] += 1
            t.update(id=state["counter"], sl0=t["sl"], opened=ts.isoformat())
            msgs.append(card(t, o, i, f, w, label))
        state["last_ts"] = ts.isoformat()
    state["trade"] = t
    li = len(df) - 1
    d = 1 if o["bullTrend"].iloc[li] else -1
    nm = factor_names(o, li)
    sc = float(o["score"].iloc[li])
    state["snapshot"] = dict(
        updated=df.index[li].isoformat(), label=label, preset=preset, price=float(df["Close"].iloc[li]),
        trend="Bullish" if d == 1 else "Bearish", score=sc, grade="A" if sc >= GRADE_A else "B",
        status=("No Trade" if not t else ("Active" if t["stage"] == 0 else f"TP{t['stage']} ✓ — Trail")),
        factors=[dict(name=nm[k], ok=bool(f[k][0 if d == 1 else 1].iloc[li]), pts=w[k]) for k in w])
    return state, msgs

# ───────────── I/O ─────────────
def fetch():
    import yfinance as yf
    period, htf_i, htf_p = FETCH[INTERVAL]
    df = yf.Ticker(SYMBOL).history(period=period, interval=INTERVAL)[["Open", "High", "Low", "Close"]].dropna()
    df = df.iloc[:-1]  # drop the still-forming candle -> closed bars only (no repaint)
    h = yf.Ticker(SYMBOL).history(period=htf_p, interval=htf_i)["Close"].dropna()
    hb = bool(h.iloc[-1] > ema(h, 50).iloc[-1])
    return df, hb, (not hb)

def send(text):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    print(text, "\n" + "-" * 30)
    if tok and chat:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=20)
        if not r.ok: print("Telegram error:", r.text)

def run_once():
    state = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
    df, hb, hs = fetch()
    state, msgs = process(df, hb, hs, state)
    for m in msgs: send(m)
    json.dump(state, open(STATE_FILE, "w"), indent=1)

if __name__ == "__main__":
    if "--loop" in sys.argv:
        while True:
            try: run_once()
            except Exception as e: print("error:", e)
            time.sleep(60)
    else:
        run_once()
