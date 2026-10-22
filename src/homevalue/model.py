"""Features, models and calibrated price ranges.

Everything a feature uses is known on the sale date, so the evaluation never peeks at the future:
  market      county median log $/sqft over the three months before the sale's month
  comp_ppsf   median log $/sqft of the 15 most recent earlier sales in the same assessor neighborhood (subarea),
              the "comparable sales" signal an agent would use; falls back to the zip code, then the county
  house       size, lot, grade (construction quality), condition, baths, bedrooms, age, views, waterfront, ...

Models: a comps baseline ($/sqft of recent neighborhood sales x this home's size), a log-linear hedonic regression,
and gradient-boosted trees, all predicting log price. Price ranges use split conformal prediction: the residual
quantile on a calibration period that follows training gives a range with a stated chance of containing the price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

COMPS = 15
NUMERIC = [
    "log_sqft",
    "log_lot",
    "grade",
    "condition",
    "baths",
    "bedrooms",
    "stories",
    "basement_sqft",
    "garage_sqft",
    "age",
    "renovated",
    "views",
    "waterfront",
    "traffic_noise",
    "market",
    "comp_ppsf",
]
CATEGORICAL = ["zip"]


def market_index(df: pd.DataFrame) -> pd.Series:
    """Median log $/sqft by calendar month."""
    lp = np.log(df.price / df.living_sqft)
    return lp.groupby(df.date.dt.to_period("M")).median()


def lagged_market(dates: pd.Series, index: pd.Series) -> np.ndarray:
    """For each date: the mean of the index over the three full months before its month."""
    out = np.empty(len(dates))
    months = dates.dt.to_period("M")
    cache: dict = {}
    for i, m in enumerate(months):
        if m not in cache:
            prior = index[(index.index < m) & (index.index >= m - 3)]
            cache[m] = float(prior.mean()) if len(prior) else float(index[index.index < m].iloc[-1:].mean())
        out[i] = cache[m]
    return out


def prior_comps(df: pd.DataFrame, key: str, k: int = COMPS) -> np.ndarray:
    """Median log $/sqft of the k most recent sales in the same `key` group strictly before each sale's date."""
    lp = np.log(df.price.to_numpy() / df.living_sqft.to_numpy())
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby(key, sort=False).indices.items():
        idx = idx[np.argsort(df.date.to_numpy()[idx], kind="stable")]
        dates = df.date.to_numpy()[idx]
        vals = lp[idx]
        for j in range(len(idx)):
            # sales strictly earlier than this one (same-day sales are not yet "known")
            n_before = np.searchsorted(dates, dates[j], side="left")
            if n_before >= 5:
                out[idx[j]] = np.median(vals[max(0, n_before - k) : n_before])
    return out


def features(df: pd.DataFrame) -> pd.DataFrame:
    idx = market_index(df)
    f = df.copy()
    f["log_sqft"] = np.log(f.living_sqft)
    f["log_lot"] = np.log(f.lot_sqft)
    f["market"] = lagged_market(f.date, idx)
    comp = prior_comps(f, "subarea")
    by_zip = prior_comps(f, "zip")
    comp = np.where(np.isnan(comp), by_zip, comp)
    f["comp_ppsf"] = np.where(np.isnan(comp), f.market, comp)
    f = f[f.market.notna()]  # the first month has no earlier months to build a market index from
    for c in ["basement_sqft", "garage_sqft", "stories", "condition", "baths", "bedrooms"]:
        f[c] = f[c].fillna(0)
    return f


def comps_baseline(f: pd.DataFrame) -> np.ndarray:
    return f.comp_ppsf.to_numpy() + f.log_sqft.to_numpy()


def hedonic():
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
        ]
    )
    return make_pipeline(pre, Ridge(alpha=1.0))


def boosted():
    pre = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return make_pipeline(
        pre,
        HistGradientBoostingRegressor(
            categorical_features=[0],
            learning_rate=0.05,
            max_iter=1500,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=0,
        ),
    )


@dataclass
class Conformal:
    """Symmetric range in log space: log price within +- q of the prediction."""

    q: dict[float, float]

    @classmethod
    def fit(cls, y_log: np.ndarray, pred_log: np.ndarray, levels=(0.8, 0.9)) -> Conformal:
        r = np.sort(np.abs(y_log - pred_log))
        n = len(r)
        return cls({lv: float(r[min(n - 1, int(np.ceil((n + 1) * lv)) - 1)]) for lv in levels})

    def interval(self, pred_log: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
        return np.exp(pred_log - self.q[level]), np.exp(pred_log + self.q[level])


def ape(price: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(pred - price) / price


def summary(price: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    e = ape(price, pred)
    return {
        "median_abs_pct_error": round(100 * float(np.median(e)), 2),
        "within_5pct": round(100 * float((e <= 0.05).mean()), 1),
        "within_10pct": round(100 * float((e <= 0.10).mean()), 1),
        "within_20pct": round(100 * float((e <= 0.20).mean()), 1),
    }


@dataclass
class MondrianConformal:
    """Conformal ranges calibrated separately within bands of *predicted* price (known when estimating), so
    expensive homes, whose prices vary more, get wider ranges than one county-wide width would give them."""

    edges: np.ndarray  # predicted-log-price cut points between bands
    q: dict[float, np.ndarray]  # level -> per-band half width in log space

    @classmethod
    def fit(cls, y_log: np.ndarray, pred_log: np.ndarray, bands: int = 4, levels=(0.8, 0.9)) -> MondrianConformal:
        edges = np.quantile(pred_log, np.linspace(0, 1, bands + 1)[1:-1])
        band = np.searchsorted(edges, pred_log)
        q = {}
        for lv in levels:
            per = []
            for b in range(bands):
                r = np.sort(np.abs(y_log - pred_log)[band == b])
                n = len(r)
                per.append(float(r[min(n - 1, int(np.ceil((n + 1) * lv)) - 1)]))
            q[lv] = np.array(per)
        return cls(edges, q)

    def interval(self, pred_log: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
        w = self.q[level][np.searchsorted(self.edges, pred_log)]
        return np.exp(pred_log - w), np.exp(pred_log + w)
