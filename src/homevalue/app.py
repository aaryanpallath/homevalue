"""A small web app: search a King County address, get an estimate with a calibrated range and the comparable sales
behind it.

    uv run uvicorn homevalue.app:app      # after `python -m homevalue.evaluate` has written data/model.joblib
"""

from __future__ import annotations

import re
from datetime import date
from functools import cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from . import model as M
from .data import homes, load

DATA = Path(__file__).resolve().parents[2] / "data"
PID = re.compile(r"^\d{6}-\d{4}$")
app = FastAPI(title="homevalue")


@cache
def state():
    bundle = joblib.load(DATA / "model.joblib")
    sales = load(DATA)
    index = M.market_index(sales)
    h = homes(DATA)
    return bundle, sales, index, h, h.address.str.upper()


def _features(home: pd.Series, sales: pd.DataFrame, index: pd.Series, today: pd.Timestamp) -> pd.DataFrame:
    market = float(M.lagged_market(pd.Series([today]), index)[0])
    lp = np.log(sales.price / sales.living_sqft)
    comp = np.nan
    for key in ("subarea", "zip"):
        recent = lp[sales[key] == home[key]].iloc[-M.COMPS :]
        if len(recent) >= 5:
            comp = float(np.median(recent))
            break
    row = home.to_dict()
    row.update(
        log_sqft=np.log(home.living_sqft),
        log_lot=np.log(home.lot_sqft),
        market=market,
        comp_ppsf=market if np.isnan(comp) else comp,
        age=today.year - home.year_built,
        renovated=int((home.year_renovated or 0) > 0),
    )
    for c in ["basement_sqft", "garage_sqft", "stories", "condition", "baths", "bedrooms"]:
        row[c] = 0 if pd.isna(row[c]) else row[c]
    return pd.DataFrame([row])[M.NUMERIC + M.CATEGORICAL]


@app.get("/")
def page():
    return FileResponse(Path(__file__).with_name("static") / "index.html")


@app.get("/api/search")
def search(q: str = Query(min_length=3, max_length=80)):
    _, _, _, h, upper = state()
    hits = h[upper.str.contains(q.upper(), regex=False)].head(10)
    return [{"pid": r.pid, "address": r.address, "zip": r.zip} for r in hits.itertuples()]


@app.get("/api/estimate/{pid}")
def estimate(pid: str):
    if not PID.match(pid):
        raise HTTPException(400, "parcel id looks like 123456-7890")
    bundle, sales, index, h, _ = state()
    found = h[h.pid == pid]
    if found.empty:
        raise HTTPException(404, "no single-family home with that parcel id")
    home = found.iloc[0]
    today = pd.Timestamp(sales.date.max()) + pd.Timedelta(days=1)
    pred_log = bundle["model"].predict(_features(home, sales, index, today))
    lo, hi = bundle["conformal"].interval(pred_log, 0.8)
    similar = sales[
        (sales.subarea == home.subarea)
        & (sales.date >= today - pd.Timedelta(days=365))
        & (sales.living_sqft.between(0.7 * home.living_sqft, 1.3 * home.living_sqft))
    ]
    comps = similar.sort_values("date", ascending=False).head(5)
    return {
        "pid": pid,
        "address": home.address,
        "zip": home.zip,
        "facts": {
            "living_sqft": int(home.living_sqft),
            "lot_sqft": int(home.lot_sqft),
            "bedrooms": None if pd.isna(home.bedrooms) else int(home.bedrooms),
            "baths": None if pd.isna(home.baths) else float(home.baths),
            "year_built": None if pd.isna(home.year_built) else int(home.year_built),
            "grade": None if pd.isna(home.grade) else int(home.grade),
        },
        "estimate": round(float(np.exp(pred_log[0])), -3),
        "range_80": [round(float(lo[0]), -3), round(float(hi[0]), -3)],
        "as_of": str(today.date()),
        "model_trained_through": bundle["trained_through"],
        "comparable_sales": [
            {
                "address": c.address,
                "date": str(c.date.date()),
                "price": int(c.price),
                "living_sqft": int(c.living_sqft),
                "price_per_sqft": round(c.price / c.living_sqft),
            }
            for c in comps.itertuples()
        ],
        "note": "Estimate from public assessor records; not an appraisal.",
        "generated": str(date.today()),
    }
