"""Nightly build: all-NSE price history -> Gann pivots, time-cycle windows and astro backtest.
Writes JSON into OUT (default ./out):
  meta.json            every stock: symbol, name, last close, change, next pivot window
  astro.json           astro events (recent + upcoming) and Gann seasonal pressure dates
  s/<SYMBOL>.json      one file per stock
"""
import datetime as dt, io, json, math, os, sys, time, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import astro

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
TODAY = dt.date.today()
HORIZON = 400                # days ahead to project
ASTRO_FROM = dt.date(2005, 1, 1)

# Gann counts (days and weeks). From Gann's Square of 144 / 1953 Master Calculator notes plus
# squares and cubes, extended to 512, 729, 1000 and 1440 as requested.
CYCLES = [30, 45, 49, 60, 63, 64, 72, 84, 90, 98, 100, 108, 120, 135, 144, 147, 180, 196, 216,
          225, 240, 270, 288, 315, 343, 360, 432, 512, 540, 576, 720, 729, 1000, 1080, 1440]
D_K, W_K = 20, 13            # swing window: daily pivots = highest/lowest within ±20 days, weekly ±13 weeks
D_YEARS = 3                  # daily pivots from the last 3 years; weekly pivots from full history
TURN_K, TURN_WIN = 10, 2     # astro backtest: turning point = ±10-day swing, hit if within ±2 trading days

# Yahoo Finance tickers for each index (first one that returns data is used)
INDEX_TICKERS = {
    "N50": ["^NSEI"], "SENSEX": ["^BSESN"], "NN50": ["^NSMIDCP", "NIFTY_NEXT_50.NS"], "N100": ["^CNX100"],
    "N200": ["^CNX200"], "N500": ["^CRSLDX"], "MID150": ["NIFTYMIDCAP150.NS", "NIFTY_MIDCAP_150.NS"],
    "SML250": ["NIFTYSMLCAP250.NS", "NIFTY_SMLCAP_250.NS"], "MIC250": ["NIFTY_MICROCAP250.NS", "NIFTYMICROCAP250.NS"],
    "BANK": ["^NSEBANK"], "PVTBANK": ["NIFTY_PVT_BANK.NS"], "PSUBANK": ["^CNXPSUBANK"], "FIN": ["NIFTY_FIN_SERVICE.NS"],
    "IT": ["^CNXIT"], "AUTO": ["^CNXAUTO"], "ENERGY": ["^CNXENERGY"], "OILGAS": ["NIFTY_OIL_AND_GAS.NS"],
    "FMCG": ["^CNXFMCG"], "PHARMA": ["^CNXPHARMA"], "HEALTH": ["NIFTY_HEALTHCARE.NS"], "METAL": ["^CNXMETAL"],
    "REALTY": ["^CNXREALTY"], "MEDIA": ["^CNXMEDIA"], "CDUR": ["NIFTY_CONSR_DURBL.NS"], "CONSUMP": ["^CNXCONSUM"],
    "INFRA": ["^CNXINFRA"], "PSE": ["^CNXPSE"], "DEFENCE": ["NIFTY_IND_DEFENCE.NS"], "CAPMKT": ["NIFTY_CAPITAL_MKT.NS"],
    "CHEM": ["NIFTY_CHEMICALS.NS"],
}

# NSE (niftyindices.com) names, used when Yahoo has no ticker for an index
NSE_INDEX_NAMES = {"MIC250": "NIFTY MICROCAP 250", "PVTBANK": "NIFTY PRIVATE BANK", "OILGAS": "NIFTY OIL & GAS",
                   "HEALTH": "NIFTY HEALTHCARE INDEX", "MEDIA": "NIFTY MEDIA", "CDUR": "NIFTY CONSUMER DURABLES",
                   "DEFENCE": "NIFTY INDIA DEFENCE", "CAPMKT": "NIFTY CAPITAL MARKETS", "CHEM": "NIFTY CHEMICALS"}
LISTS = json.load(open(os.path.join(os.path.dirname(__file__), "index_lists.json")))

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


def log(*a):
    print(*a, flush=True)


def nse_symbols():
    """All NSE equities (EQ/BE series). Falls back to the screener universe."""
    try:
        req = urllib.request.Request("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv", headers=UA)
        df = pd.read_csv(io.BytesIO(urllib.request.urlopen(req, timeout=60).read()))
        df.columns = [c.strip() for c in df.columns]
        df = df[df["SERIES"].str.strip().isin(["EQ", "BE"])]
        out = {r["SYMBOL"].strip(): r["NAME OF COMPANY"].strip() for _, r in df.iterrows()}
        log("NSE list:", len(out))
        if len(out) > 1000:
            return out
    except Exception as e:
        log("NSE list failed:", e)
    fb = os.path.join(os.path.dirname(__file__), "symbols.txt")
    return {l.split("|")[0]: l.split("|")[1].strip() for l in open(fb) if "|" in l}


def history(syms):
    import yfinance as yf
    tick = [s + ".NS" for s in syms]
    for attempt in range(3):
        try:
            df = yf.download(tick, period="max", interval="1d", group_by="ticker", auto_adjust=False,
                             threads=True, progress=False, timeout=60)
            break
        except Exception as e:
            log("batch retry", attempt, e)
            time.sleep(20)
    else:
        return {}
    res = {}
    for s, t in zip(syms, tick):
        try:
            d = df[t] if len(tick) > 1 else df
            d = d[["Open", "High", "Low", "Close"]].dropna()
            d = d[(d["High"] > 0) & (d["Low"] > 0)]
            if len(d) >= 60:
                d.index = pd.to_datetime(d.index).tz_localize(None)
                res[s] = d
        except Exception:
            pass
    return res


def swings(h, l, k, confirm=True):
    """Indices of swing highs / lows: extreme within ±k bars."""
    n = len(h)
    hi, lo = [], []
    hv, lv = h.values, l.values
    for i in range(n):
        a, b = max(0, i - k), min(n, i + k + 1)
        if confirm and i + k >= n:
            break
        if hv[i] == hv[a:b].max():
            hi.append(i)
        if lv[i] == lv[a:b].min():
            lo.append(i)
    return hi, lo


def index_history(cands):
    import yfinance as yf
    for t in cands:
        try:
            d = yf.download(t, period="max", interval="1d", auto_adjust=False, progress=False, timeout=60)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d[["Open", "High", "Low", "Close"]].dropna()
            d = d[(d["High"] > 0) & (d["Low"] > 0)]
            if len(d) >= 60:
                d.index = pd.to_datetime(d.index).tz_localize(None)
                return t, d
        except Exception as e:
            log("index fail", t, e)
    return None, None


def nse_index_history(name):
    """Daily OHLC from niftyindices.com historical data, one year per request."""
    rows = []
    url = "https://www.niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString"
    hdr = {**UA, "Content-Type": "application/json; charset=UTF-8", "X-Requested-With": "XMLHttpRequest",
           "Referer": "https://www.niftyindices.com/reports/historical-data", "Origin": "https://www.niftyindices.com"}
    for y in range(2005, TODAY.year + 1):
        a = dt.date(y, 1, 1).strftime("%d-%b-%Y")
        b = min(dt.date(y, 12, 31), TODAY).strftime("%d-%b-%Y")
        body = json.dumps({"cinfo": "{'name':'%s','startDate':'%s','endDate':'%s','indexName':'%s'}" % (name, a, b, name)}).encode()
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, data=body, headers=hdr), timeout=60)
            for x in json.loads(json.loads(r.read())["d"]):
                rows.append((pd.to_datetime(x["HistoricalDate"]), float(x["OPEN"]), float(x["HIGH"]), float(x["LOW"]), float(x["CLOSE"])))
        except Exception as e:
            if y == TODAY.year:
                log("niftyindices fail", name, y, e)
        time.sleep(0.5)
    if len(rows) < 60:
        return None
    d = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"]).drop_duplicates("Date").set_index("Date").sort_index()
    return d[(d["High"] > 0) & (d["Low"] > 0)]


def equal_weight(members, keep):
    """Equal-weight index from member stocks (base 1000). Only the timing of turns is meaningful, not the level."""
    fr = [keep[m] for m in members if m in keep]
    if len(fr) < 5:
        return None
    c = pd.concat({i: f["Close"] for i, f in enumerate(fr)}, axis=1)
    h = pd.concat({i: f["High"] for i, f in enumerate(fr)}, axis=1)
    l = pd.concat({i: f["Low"] for i, f in enumerate(fr)}, axis=1)
    pc = c.shift(1)
    rc, rh, rl = (c / pc).mean(axis=1), (h / pc).mean(axis=1), (l / pc).mean(axis=1)
    ok = c.notna().sum(axis=1) >= 3
    rc, rh, rl = rc[ok].iloc[1:], rh[ok].iloc[1:], rl[ok].iloc[1:]
    close = 1000 * rc.cumprod()
    prev = close.shift(1).fillna(1000)
    d = pd.DataFrame({"Open": prev, "High": prev * rh, "Low": prev * rl, "Close": close}).dropna()
    return d if len(d) >= 250 else None


def record(s, name, d, ev_hist):
    piv = pivots(d)
    win = windows(piv)
    st = astro_stats(d, ev_hist)
    last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
    major = [p for p in piv if p[1] in ("W", "ATH", "ATL", "52W")][-12:]
    rec = {"s": s, "n": name, "asof": d.index[-1].date().isoformat(), "last": round(last, 2),
           "first": d.index[0].date().isoformat(),
           "piv": [[p[0], p[1], p[2].isoformat(), round(p[3], 2)] for p in major],
           "win": win, "astro": st}
    with open(os.path.join(OUT, "s", f"{s}.json"), "w") as f:
        json.dump(rec, f, separators=(",", ":"))
    nxt = next((w for w in win if w["b"] >= TODAY.isoformat()), None)
    return [s, name, round(last, 2), round(last / prev - 1, 4) if prev else 0,
            nxt["a"] if nxt else "", nxt["b"] if nxt else "", nxt["sc"] if nxt else 0]


def pivots(d):
    w = d.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    out = []
    wh, wl = swings(w["High"], w["Low"], W_K)
    for i in wh:  # weekly swing: use the actual day of the extreme inside that week
        wk = d.loc[w.index[i] - pd.Timedelta(days=6):w.index[i]]
        out.append(("H", "W", wk["High"].idxmax().date(), float(wk["High"].max())))
    for i in wl:
        wk = d.loc[w.index[i] - pd.Timedelta(days=6):w.index[i]]
        out.append(("L", "W", wk["Low"].idxmin().date(), float(wk["Low"].min())))
    recent = d[d.index >= pd.Timestamp(TODAY - dt.timedelta(days=365 * D_YEARS))]
    if len(recent) > 2 * D_K:
        dh, dl = swings(recent["High"], recent["Low"], D_K)
        out += [("H", "D", recent.index[i].date(), float(recent["High"].iloc[i])) for i in dh]
        out += [("L", "D", recent.index[i].date(), float(recent["Low"].iloc[i])) for i in dl]
    # all-time and 52-week extremes always count
    out.append(("H", "ATH", d["High"].idxmax().date(), float(d["High"].max())))
    out.append(("L", "ATL", d["Low"].idxmin().date(), float(d["Low"].min())))
    y = d[d.index >= pd.Timestamp(TODAY - dt.timedelta(days=365))]
    if len(y):
        out.append(("H", "52W", y["High"].idxmax().date(), float(y["High"].max())))
        out.append(("L", "52W", y["Low"].idxmin().date(), float(y["Low"].min())))
    # de-duplicate same date & side, keep the strongest label
    rank = {"ATH": 4, "ATL": 4, "W": 3, "52W": 2, "D": 1}
    best = {}
    for p in out:
        k = (p[0], p[2])
        if k not in best or rank[p[1]] > rank[best[k][1]]:
            best[k] = p
    return sorted(best.values(), key=lambda p: p[2])


WEIGHT = {"ATH": 3, "ATL": 3, "W": 2, "52W": 2, "D": 1}


def windows(piv):
    lo, hi = TODAY - dt.timedelta(days=10), TODAY + dt.timedelta(days=HORIZON)
    hits = []
    for side, tf, d0, px in piv:
        for c in CYCLES:
            for unit, days in (("d", c), ("w", c * 7)):
                t = d0 + dt.timedelta(days=days)
                if lo <= t <= hi:
                    hits.append((t, WEIGHT[tf] * (1.5 if unit == "w" else 1), f"{c}{unit}", side, tf, d0))
    hits.sort()
    wins, i = [], 0
    while i < len(hits):
        j = i
        while j + 1 < len(hits) and (hits[j + 1][0] - hits[i][0]).days <= 4:
            j += 1
        grp = hits[i:j + 1]
        score = sum(h[1] for h in grp)
        if len(grp) >= 3:
            ds = [h[0] for h in grp]
            mid = ds[0] + (ds[-1] - ds[0]) / 2
            top = sorted(grp, key=lambda h: -h[1])[:6]
            wins.append({"a": ds[0].isoformat(), "b": ds[-1].isoformat(), "m": mid.isoformat(), "sc": round(score, 1),
                         "n": len(grp), "why": [[h[2], h[3], h[4], h[5].isoformat()] for h in top]})
        i = j + 1
    # keep the strongest windows, shown in date order
    strong = sorted(wins, key=lambda w: -w["sc"])[:14]
    return sorted(strong, key=lambda w: w["a"])


def astro_stats(d, ev):
    """For each event category: how often a ±TURN_K-day turning point fell within ±TURN_WIN trading days."""
    d = d[d.index >= pd.Timestamp(ASTRO_FROM)]
    if len(d) < 250:
        return {}
    hi, lo = swings(d["High"], d["Low"], TURN_K)
    turn = np.zeros(len(d), bool)
    turn[hi] = True
    turn[lo] = True
    near = np.convolve(turn.astype(int), np.ones(2 * TURN_WIN + 1, int), "same") > 0
    valid = len(d) - TURN_K  # last bars cannot be confirmed yet
    base = near[:valid].mean()
    c = d["Close"].values
    mv = np.full(len(d), np.nan)
    mv[:-5] = np.abs(c[5:] / c[:-5] - 1)
    mv_all = np.nanmean(mv[:valid])
    idx = d.index.values.astype("datetime64[D]")
    res = {}
    for e in ev:
        t = np.datetime64(e["d"])
        if t < idx[0] or t > idx[valid - 1]:
            continue
        i = int(np.searchsorted(idx, t))
        if i >= valid:
            continue
        r = res.setdefault(e["c"], [0, 0, []])
        r[0] += 1
        r[1] += int(near[i])
        if not np.isnan(mv[i]):
            r[2].append(mv[i])
    out = {}
    for k, (n, h, m) in res.items():
        if n >= 5:
            out[k] = [n, h, round(float(base), 3), round(float(np.mean(m)) if m else 0, 4), round(float(mv_all), 4)]
    return out


def seasonal(year_from, year_to):
    """Gann seasonal pressure dates: equinoxes, solstices and the midpoints between them."""
    out = []
    for y in range(year_from, year_to + 1):
        for m, d, t in ((2, 4, "Gann seasonal (Feb 4)"), (3, 21, "Spring equinox"), (5, 6, "Gann seasonal (May 6)"),
                        (6, 21, "Summer solstice"), (8, 8, "Gann seasonal (Aug 8)"), (9, 23, "Autumn equinox"),
                        (11, 8, "Gann seasonal (Nov 8)"), (12, 21, "Winter solstice")):
            out.append({"d": dt.date(y, m, d).isoformat(), "c": "Gann pressure date", "t": t})
    return out


def main():
    os.makedirs(os.path.join(OUT, "s"), exist_ok=True)
    log("astro events…")
    ev = astro.events(ASTRO_FROM, TODAY + dt.timedelta(days=HORIZON))
    ev_hist = [e for e in ev if e["d"] <= TODAY.isoformat()]
    up = [e for e in ev if (TODAY - dt.timedelta(days=30)).isoformat() <= e["d"]]
    press = [e for e in seasonal(TODAY.year - 1, TODAY.year + 2) if (TODAY - dt.timedelta(days=30)).isoformat() <= e["d"]]
    json.dump({"asof": TODAY.isoformat(), "events": up, "pressure": press}, open(os.path.join(OUT, "astro.json"), "w"), separators=(",", ":"))

    idx_rows = []
    for code, cands in INDEX_TICKERS.items():
        t, d = index_history(cands)
        log("index", code, t, None if d is None else len(d))
        if d is not None:
            try:
                row = record("IDX_" + code, code, d, ev_hist)
                idx_rows.append(row + [t])
            except Exception as e:
                log("index record fail", code, e)
    have = {r[1] for r in idx_rows}
    missing = [c for c in INDEX_TICKERS if c not in have]
    for code in list(missing):
        if code in NSE_INDEX_NAMES:
            d = nse_index_history(NSE_INDEX_NAMES[code])
            log("niftyindices", code, None if d is None else len(d))
            if d is not None:
                try:
                    idx_rows.append(record("IDX_" + code, code, d, ev_hist) + ["NSE"])
                    missing.remove(code)
                except Exception as e:
                    log("index record fail", code, e)
    NEEDED = {s for c in missing for s in LISTS.get(c, [])}
    keep = {}
    names = nse_symbols()
    syms = sorted(names)
    meta = []
    B = 40
    for b in range(0, len(syms), B):
        batch = syms[b:b + B]
        hist = history(batch)
        log(f"{b + len(batch)}/{len(syms)} got {len(hist)}")
        for s, d in hist.items():
            if s in NEEDED:
                keep[s] = d[d.index >= pd.Timestamp(ASTRO_FROM)]
            try:
                meta.append(record(s, names.get(s, s), d, ev_hist))
            except Exception as e:
                log("fail", s, e)
        time.sleep(1.5)
    for code in missing:
        d = equal_weight(LISTS.get(code, []), keep)
        log("equal-weight", code, None if d is None else len(d))
        if d is not None:
            try:
                idx_rows.append(record("IDX_" + code, code, d, ev_hist) + ["EQW"])
            except Exception as e:
                log("index record fail", code, e)
    json.dump({"asof": TODAY.isoformat(), "cols": ["sym", "name", "last", "chg", "winA", "winB", "winScore"], "rows": meta, "idx": idx_rows},
              open(os.path.join(OUT, "meta.json"), "w"), separators=(",", ":"))
    log("done", len(meta), "stocks")
    if len(meta) < 200:
        sys.exit("too few stocks fetched; not publishing")


if __name__ == "__main__":
    main()
