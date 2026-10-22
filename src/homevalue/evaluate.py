"""Time-honest evaluation, then the model the app serves.

    uv run python -m homevalue.evaluate      # results/eval.json, results/*.png, data/model.joblib

Train on sales before October 2025, calibrate price ranges on October 2025 - March 2026, and test on April -
September 2026: every test sale happens after everything the model or the ranges were fit on. The served model is
then refit the same way one period later (train through March 2026, calibrate on the test period), so its ranges
come from the most recent six months.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import model as M  # noqa: E402
from .data import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
X = M.NUMERIC + M.CATEGORICAL
SPLITS = ("2025-10-01", "2026-04-01")


def bootstrap_diff(price: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0) -> list[float]:
    """95% interval for median APE(a) - median APE(b), resampling test sales."""
    rng = np.random.default_rng(seed)
    ea, eb = M.ape(price, a), M.ape(price, b)
    d = [np.median(ea[i]) - np.median(eb[i]) for i in (rng.integers(0, len(price), len(price)) for _ in range(n))]
    return [round(100 * float(v), 2) for v in np.quantile(d, [0.025, 0.975])]


def main() -> None:
    f = M.features(load(ROOT / "data"))
    train, cal, test = f[f.date < SPLITS[0]], f[(f.date >= SPLITS[0]) & (f.date < SPLITS[1])], f[f.date >= SPLITS[1]]
    y = test.price.to_numpy()
    preds = {"comps ($/sqft of recent neighborhood sales)": np.exp(M.comps_baseline(test))}
    h = M.hedonic().fit(train[X], np.log(train.price))
    preds["hedonic regression"] = np.exp(h.predict(test[X]))
    g = M.boosted().fit(train[X], np.log(train.price))
    preds["gradient boosting"] = np.exp(g.predict(test[X]))

    cf = M.Conformal.fit(np.log(cal.price.to_numpy()), g.predict(cal[X]))
    pt = g.predict(test[X])
    tiers = pd.qcut(test.price, 4, labels=["lowest quarter", "second", "third", "highest quarter"])
    mc = M.MondrianConformal.fit(np.log(cal.price.to_numpy()), g.predict(cal[X]))
    coverage = {}
    for name, conf in (("one width for every home", cf), ("width per predicted-price band", mc)):
        coverage[name] = {}
        for lv in (0.8, 0.9):
            lo, hi = conf.interval(pt, lv)
            inside = (y >= lo) & (y <= hi)
            coverage[name][str(lv)] = {
                "overall": round(float(inside.mean()), 3),
                "median_range_half_width_pct": round(float(100 * np.median((hi - lo) / 2 / np.exp(pt))), 1),
                "by_price_quarter": {
                    str(t): round(float(inside[(tiers == t).to_numpy()].mean()), 3) for t in tiers.cat.categories
                },
            }
    gb = preds["gradient boosting"]
    by_tier = {str(t): M.summary(y[(tiers == t).to_numpy()], gb[(tiers == t).to_numpy()]) for t in tiers.cat.categories}
    out = {
        "data": {
            "sales": len(f),
            "train": [str(train.date.min().date()), str(train.date.max().date()), len(train)],
            "calibration": [str(cal.date.min().date()), str(cal.date.max().date()), len(cal)],
            "test": [str(test.date.min().date()), str(test.date.max().date()), len(test)],
        },
        "models": {k: M.summary(y, v) for k, v in preds.items()},
        "boosting_minus_hedonic_median_error_ci95": bootstrap_diff(
            y, preds["gradient boosting"], preds["hedonic regression"]
        ),
        "boosting_by_price_quarter": by_tier,
        "price_ranges": coverage,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval.json").write_text(json.dumps(out, indent=2) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, p in preds.items():
        e = np.sort(M.ape(y, p))
        ax[0].plot(100 * e, np.arange(1, len(e) + 1) / len(e), label=name)
    ax[0].set(
        xlim=(0, 40),
        xlabel="absolute error (% of sale price)",
        ylabel="share of test sales",
        title=f"How close each model gets ({len(test):,} sales, Apr-Sep 2026)",
    )
    ax[0].legend(fontsize=8)
    q = list(by_tier)
    ax[1].bar(q, [by_tier[t]["median_abs_pct_error"] for t in q], color="#2f6fdb")
    ax[1].set(ylabel="median absolute error (%)", title="Gradient boosting by price quarter")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "accuracy.png", dpi=130)

    # the served model: same recipe, one period later
    served_train, served_cal = f[f.date < SPLITS[1]], f[f.date >= SPLITS[1]]
    gs = M.boosted().fit(served_train[X], np.log(served_train.price))
    cfs = M.MondrianConformal.fit(np.log(served_cal.price.to_numpy()), gs.predict(served_cal[X]))
    joblib.dump(
        {
            "model": gs,
            "conformal": cfs,
            "trained_through": str(served_train.date.max().date()),
            "calibrated_on": [str(served_cal.date.min().date()), str(served_cal.date.max().date())],
        },
        ROOT / "data" / "model.joblib",
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
