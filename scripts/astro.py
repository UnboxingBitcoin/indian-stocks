"""Astro events with Swiss Ephemeris (Moshier, no data files needed).
Sidereal positions use Lahiri, matching the Krishna Mukh astro calendar.
Times are converted to IST dates."""
import datetime as dt
import swisseph as swe

FLG = swe.FLG_MOSEPH | swe.FLG_SPEED
SID = FLG | swe.FLG_SIDEREAL
swe.set_sid_mode(swe.SIDM_LAHIRI)
IST = 5.5 / 24

P = {"Sun": swe.SUN, "Moon": swe.MOON, "Mercury": swe.MERCURY, "Venus": swe.VENUS, "Mars": swe.MARS,
     "Jupiter": swe.JUPITER, "Saturn": swe.SATURN, "Uranus": swe.URANUS, "Neptune": swe.NEPTUNE,
     "Pluto": swe.PLUTO, "Rahu": swe.MEAN_NODE}
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
         "Sagittarius", "Capricorn", "Aquarius", "Pisces"]


def jd(d):
    return swe.julday(d.year, d.month, d.day, 0.0)


def jd2date(j):
    y, m, d, h = swe.revjul(j + IST)
    return dt.date(y, m, d)


def lon(p, j, sid=False):
    return swe.calc_ut(j, P[p], SID if sid else FLG)[0][0]


def spd(p, j):
    return swe.calc_ut(j, P[p], FLG)[0][3]


def root(f, a, b, it=40):
    fa = f(a)
    for _ in range(it):
        m = (a + b) / 2
        fm = f(m)
        if (fa < 0) == (fm < 0):
            a, fa = m, fm
        else:
            b = m
    return (a + b) / 2


def wrap(x):
    return (x + 180) % 360 - 180


def events(start, end):
    j0, j1 = jd(start), jd(end)
    ev = []

    def add(j, cat, label):
        ev.append({"d": jd2date(j).isoformat(), "c": cat, "t": label})

    # lunations: Moon - Sun elongation 0 (new) and 180 (full)
    step = 0.5
    j = j0
    while j < j1:
        for tgt, cat in ((0, "New moon"), (180, "Full moon")):
            f = lambda x: wrap(lon("Moon", x) - lon("Sun", x) - tgt)
            if f(j) < 0 <= f(j + step) and abs(f(j)) < 30:
                add(root(f, j, j + step), cat, cat)
        j += step
    # eclipses
    j = j0
    while True:
        r = swe.sol_eclipse_when_glob(j, swe.FLG_MOSEPH)
        t = r[1][0]
        if t > j1:
            break
        kind = "Total" if r[0] & swe.ECL_TOTAL else "Annular" if r[0] & swe.ECL_ANNULAR else "Hybrid" if r[0] & swe.ECL_ANNULAR_TOTAL else "Partial"
        add(t, "Solar eclipse", f"{kind} solar eclipse")
        j = t + 20
    j = j0
    while True:
        r = swe.lun_eclipse_when(j, swe.FLG_MOSEPH)
        t = r[1][0]
        if t > j1:
            break
        kind = "Total" if r[0] & swe.ECL_TOTAL else "Partial" if r[0] & swe.ECL_PARTIAL else "Penumbral"
        add(t, "Lunar eclipse", f"{kind} lunar eclipse")
        j = t + 20
    # stations
    for p in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"):
        j = j0
        while j < j1:
            a, b = spd(p, j), spd(p, j + 1)
            if (a < 0) != (b < 0):
                t = root(lambda x: spd(p, x), j, j + 1)
                cat = f"{p} station" if p in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn") else "Outer planet station"
                add(t, cat, f"{p} turns {'retrograde' if b < 0 else 'direct'}")
            j += 1
    # sidereal ingress of slow planets
    for p in ("Mars", "Jupiter", "Saturn", "Rahu", "Uranus", "Neptune", "Pluto"):
        j = j0
        s0 = int(lon(p, j, True) // 30)
        while j < j1:
            s1 = int(lon(p, j + 1, True) // 30)
            if s1 != s0:
                bound = (s1 if (s1 - s0) % 12 == 1 else s0) * 30
                t = root(lambda x: wrap(lon(p, x, True) - bound), j, j + 1)
                cat = f"{p} ingress" if p in ("Mars", "Jupiter", "Saturn", "Rahu") else "Outer planet ingress"
                add(t, cat, f"{p} enters {SIGNS[s1]} (sidereal)")
                s0 = s1
            j += 1
    # major aspects between slow movers (geocentric, exact)
    slow = ["Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Rahu"]
    names = {0: "conjunct", 60: "sextile", 90: "square", 120: "trine", 180: "opposite"}
    for i, p in enumerate(slow):
        for q in slow[i + 1:]:
            stp = 1 if p == "Mars" else 2
            for a in names:
                sgn = [1] if a in (0, 180) else [1, -1]
                for s in sgn:
                    f = lambda x: wrap(lon(p, x) - lon(q, x) - s * a)
                    j = j0
                    prev = f(j)
                    while j < j1:
                        cur = f(j + stp)
                        if (prev < 0) != (cur < 0) and abs(prev - cur) < 90:
                            t = root(f, j, j + stp)
                            pair = "Mars aspect" if p == "Mars" else f"{p}–{q} aspect"
                            add(t, pair, f"{p} {names[a]} {q}")
                        prev = cur
                        j += stp
    ev.sort(key=lambda e: e["d"])
    # drop duplicate lunar-phase detections
    out, seen = [], set()
    for e in ev:
        k = (e["d"], e["t"])
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


if __name__ == "__main__":
    import collections, sys
    ev = events(dt.date(2024, 1, 1), dt.date(2026, 12, 31))
    print(collections.Counter(e["c"] for e in ev))
    for e in ev:
        if any(k in e["t"] for k in ("eclipse", "Mercury", "Jupiter enters", "Saturn enters", "Jupiter conjunct", "Saturn conjunct Neptune")):
            print(e["d"], e["t"])
