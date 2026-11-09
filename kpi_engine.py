"""KPI engine: data (Modules 3), KPI calculation (4), anomaly detection + forecasting (6)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

CSV = Path(__file__).parent / "data" / "business_data.csv"
KPIS = {"revenue": "Sales revenue", "profit": "Profit", "customers": "Customers",
        "employee_score": "Employee performance", "efficiency": "Operational efficiency"}
REGIONS = {"North": 1.0, "South": 0.85, "East": 1.15, "West": 0.95}
CATEGORIES = {"Electronics": 1.3, "Apparel": 0.8, "Services": 1.0}


# ---------- Module 3: dataset ----------
def generate(path=CSV, seed=42):
    """Synthetic 36-month enterprise dataset with a few injected business shocks."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-01-01", periods=36, freq="MS")
    rows = []
    for r, rb in REGIONS.items():
        for c, cb in CATEGORIES.items():
            for i, m in enumerate(months):
                season = 1 + 0.18 * np.sin((m.month - 3) / 12 * 2 * np.pi)
                rev = 60000 * rb * cb * season * (1 + 0.012 * i) * rng.normal(1, 0.03)
                cost_ratio = rng.normal(0.72, 0.012)
                emp = 76 + 0.08 * i + rng.normal(0, 1.2)
                eff = 81 + rng.normal(0, 1.0)
                if (r, c, i) == ("East", "Electronics", 20): rev *= 0.55          # supply disruption
                if (r, c, i) == ("West", "Apparel", 12): rev *= 1.6               # promo spike
                if (c, i) == ("Services", 28): cost_ratio += 0.18                 # cost overrun
                if (r, i) == ("North", 34): emp -= 12; eff -= 9                   # attrition / downtime
                cost = rev * cost_ratio
                rows.append([m, r, c, rev, cost, rev - cost, rev / rng.normal(210, 6), emp, eff])
    cols = ["date", "region", "category", "revenue", "cost", "profit", "customers", "employee_score", "efficiency"]
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=cols).round(2).to_csv(path, index=False)


def load():
    if not CSV.exists():
        generate()
    df = pd.read_csv(CSV, parse_dates=["date"]).drop_duplicates().dropna()
    return df

DF = load()


def subset(regions=None, categories=None, pairs=None, year=None, start=None, end=None, rmin=None, rmax=None):
    """Apply slicer selections. Every slicer is AND-ed; an empty/None selection means 'no filter'."""
    d = DF
    if regions: d = d[d.region.isin(regions)]
    if categories: d = d[d.category.isin(categories)]
    if pairs: d = d[(d.region + "|" + d.category).isin(pairs)]          # hierarchy slicer: Region > Category
    if year: d = d[d.date.dt.year == int(year)]
    if start: d = d[d.date >= pd.Timestamp(start + "-01")]               # date-range slicer (YYYY-MM)
    if end: d = d[d.date <= pd.Timestamp(end + "-01")]
    if rmin is not None: d = d[d.revenue >= rmin]                         # numeric-range slicer
    if rmax is not None: d = d[d.revenue <= rmax]
    return d


def meta():
    """Option lists and bounds that the slicers are built from."""
    step = 1000
    return dict(regions=sorted(DF.region.unique()), categories=sorted(DF.category.unique()),
                years=sorted(int(y) for y in DF.date.dt.year.unique()),
                months=[x.strftime("%Y-%m") for x in sorted(DF.date.unique())],
                rev_min=int(DF.revenue.min() // step * step), rev_max=int(-(-DF.revenue.max() // step) * step), rev_step=step)


# ---------- Module 4: KPIs ----------
def monthly(d):
    g = d.groupby("date")
    return pd.DataFrame({"revenue": g.revenue.sum(), "profit": g.profit.sum(), "customers": g.customers.sum(),
                         "employee_score": g.employee_score.mean(), "efficiency": g.efficiency.mean()})


def _pct(a, b):
    return float((a / b - 1) * 100) if b else None


def kpi_cards(m):
    """Value plus MoM / QoQ / YoY; a comparison is None when the selection has too little history."""
    cards, n = {}, len(m)
    for k in KPIS:
        s = m[k]
        cards[k] = dict(label=KPIS[k], value=float(s.iloc[-1]),
                        mom=_pct(s.iloc[-1], s.iloc[-2]) if n >= 2 else None,
                        qoq=_pct(s.iloc[-3:].mean(), s.iloc[-6:-3].mean()) if n >= 6 else None,
                        yoy=_pct(s.iloc[-1], s.iloc[-13]) if n >= 13 else None)
    return cards


# ---------- Module 6: AI/ML ----------
def _X(dates, t0=0):
    t = np.arange(len(dates)) + t0
    return np.column_stack([t, np.eye(12)[dates.month - 1]])   # linear trend + month-of-year seasonality


def fit(s):
    model = LinearRegression(fit_intercept=False).fit(_X(s.index), s.values)
    resid = s.values - model.predict(_X(s.index))
    return model, resid


def forecast(s, n=6):
    model, resid = fit(s)
    future = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=n, freq="MS")
    pred = model.predict(_X(future, len(s)))
    band = 1.28 * resid.std()   # ~80% interval
    return dict(dates=[x.strftime("%Y-%m") for x in future], values=pred.round(2).tolist(),
                lo=(pred - band).round(2).tolist(), hi=(pred + band).round(2).tolist())


def _driver(d, dt, kpi, up):
    """Explain WHY: which region/category moved most away from its own normal level that month."""
    p = d.pivot_table(index="date", columns=["region", "category"], values=kpi, aggfunc="sum")
    dev = p.loc[dt] - p.median()
    r, c = dev.idxmax() if up else dev.idxmin()
    return f"{r} / {c}"


def anomalies(m, d, kpi):
    s = m[kpi]
    _, resid = fit(s)
    z = (resid - resid.mean()) / resid.std()
    iso = IsolationForest(contamination=0.05, random_state=0).fit(resid.reshape(-1, 1))
    flag = (iso.predict(resid.reshape(-1, 1)) == -1) & (np.abs(z) > 2.8)
    out = []
    for i in np.where(flag)[0]:
        dt = s.index[i]
        expected = s.iloc[i] - resid[i]
        out.append(dict(kpi=kpi, label=KPIS[kpi], date=dt.strftime("%Y-%m"), value=float(s.iloc[i]),
                        deviation=float(resid[i] / expected * 100), driver=_driver(d, dt, kpi, resid[i] > 0)))
    return out


MIN_MONTHS = 12   # forecast / anomaly model needs a full seasonal cycle


def insights(m, cards, fc, anoms, d):
    out = []
    for basis, name in (("yoy", "year-on-year"), ("qoq", "quarter-on-quarter"), ("mom", "month-on-month")):
        avail = {k: c[basis] for k, c in cards.items() if c[basis] is not None}
        if avail:
            best, worst = max(avail, key=avail.get), min(avail, key=avail.get)
            out.append(f"{cards[best]['label']} is the strongest KPI {name} ({avail[best]:+.1f}%); "
                       f"{cards[worst]['label']} is the weakest ({avail[worst]:+.1f}%).")
            break
    mg_now = m.profit.iloc[-1] / m.revenue.iloc[-1] * 100
    if len(m) >= 13:
        mg_ago = m.profit.iloc[-13] / m.revenue.iloc[-13] * 100
        out.append(f"Profit margin is {mg_now:.1f}%, {'up' if mg_now >= mg_ago else 'down'} {abs(mg_now - mg_ago):.1f} pts on a year ago.")
    else:
        out.append(f"Profit margin is {mg_now:.1f}% in the latest month of this selection.")
    if fc:
        nxt, prev = np.mean(fc["revenue"]["values"]), m.revenue.iloc[-6:].mean()
        out.append(f"Revenue is forecast to average {nxt/1e3:,.0f}K per month over the next 6 months, "
                   f"{(nxt/prev-1)*100:+.1f}% versus the last 6 months.")
        recent = [a for a in anoms if a["date"] >= m.index[-3].strftime("%Y-%m")]
        for a in recent:
            out.append(f"Alert: {a['label']} in {a['date']} was {abs(a['deviation']):.0f}% "
                       f"{'above' if a['deviation'] > 0 else 'below'} expected, driven mainly by {a['driver']}.")
        if not recent:
            out.append("No anomalies detected in the last 3 months.")
    else:
        out.append(f"Forecast and anomaly detection need at least {MIN_MONTHS} months of history; widen the date slicers.")
    if d.region.nunique() > 1 and len(m) >= 3:
        q = d[d.date >= m.index[-3]].groupby("region").revenue.sum()
        out.append(f"{q.idxmax()} leads revenue over the last quarter ({q.max()/q.sum()*100:.0f}% share); {q.idxmin()} trails.")
    return out


def dashboard(**filters):
    d = subset(**filters)
    if d.empty:
        return dict(empty=True)
    m = monthly(d)
    cards = kpi_cards(m)
    ok = len(m) >= MIN_MONTHS
    fc = {k: forecast(m[k]) if ok else None for k in KPIS}
    an = sorted([a for k in KPIS for a in anomalies(m, d, k)], key=lambda a: a["date"], reverse=True) if ok else []
    recent = d[d.date >= m.index[-12]] if ok else d
    return dict(empty=False, cards=cards, forecast=fc, anomalies=an, rows=int(len(d)), months=int(len(m)),
                series=dict(dates=[x.strftime("%Y-%m") for x in m.index], **{k: m[k].round(2).tolist() for k in KPIS}),
                by_region=recent.groupby("region").revenue.sum().round(0).to_dict(),
                by_category=recent.groupby("category").revenue.sum().round(0).to_dict(),
                insights=insights(m, cards, fc if ok else None, an, d))
