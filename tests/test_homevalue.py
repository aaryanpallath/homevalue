import csv

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from homevalue import app as webapp
from homevalue import model as M
from homevalue.data import VIEWS, build

SALE_COLS = [
    "ExciseTaxNbr",
    "Major",
    "Minor",
    "DocumentDate",
    "SalePrice",
    "PropertyType",
    "PrincipalUse",
    "SaleInstrument",
    "SaleReason",
    "SaleWarning",
]
BLDG_COLS = [
    "Major",
    "Minor",
    "BldgNbr",
    "NbrLivingUnits",
    "Address",
    "ZipCode",
    "Stories",
    "BldgGrade",
    "SqFtTotLiving",
    "SqFtTotBasement",
    "SqFtGarageAttached",
    "Bedrooms",
    "BathHalfCount",
    "Bath3qtrCount",
    "BathFullCount",
    "YrBuilt",
    "YrRenovated",
    "Condition",
]
PARCEL_COLS = ["Major", "Minor", "Area", "SubArea", "SqFtLot", "WfntLocation", "TrafficNoise", *VIEWS]


def write(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(cols)
        w.writerows(rows)


def sale(tax, minor, date, price, ptype="11", instrument="2", reason="1", warning=" "):
    return [tax, "123456", minor, date, price, ptype, "6", instrument, reason, warning]


def test_cleaning_keeps_only_arms_length_single_family_sales(tmp_path):
    write(
        tmp_path / "EXTR_RPSale.csv",
        SALE_COLS,
        [
            sale("1", "0010", "03/02/2021", "800000"),  # kept: 2020+ warranty deed (code 2)
            sale("2", "0020", "05/10/2019", "700000", instrument="3"),  # kept: statutory warranty deed
            sale("3", "0030", "06/01/2021", "650000", instrument="15"),  # quit claim deed
            sale("4", "0040", "06/01/2021", "650000", reason="11"),  # divorce settlement
            sale("5", "0050", "06/01/2021", "650000", warning="15 "),  # no market exposure
            sale("6", "0060", "06/01/2021", "1300000"),
            sale("6", "0070", "06/01/2021", "1300000"),  # two parcels, one sale
            sale("7", "0080", "06/01/2021", "40000"),  # below $50k
            sale("8", "0010", "04/01/2021", "900000"),  # same house 30 days later: dropped as a flip
        ],
    )
    rows = []
    for minor in ["0010", "0020", "0030", "0040", "0050", "0060", "0070", "0080"]:
        rows.append(
            [
                "123456",
                minor,
                "1",
                "1",
                f"1 MAIN ST {minor}",
                "98103",
                "2",
                "8",
                "2000",
                "0",
                "400",
                "3",
                "1",
                "0",
                "2",
                "1990",
                "0",
                "3",
            ]
        )
    write(tmp_path / "EXTR_ResBldg.csv", BLDG_COLS, rows)
    write(
        tmp_path / "EXTR_Parcel.csv",
        PARCEL_COLS,
        [
            ["123456", m, "10", "3", "5000", "0", "0", *(["0"] * len(VIEWS))]
            for m in ["0010", "0020", "0030", "0040", "0050", "0060", "0070", "0080"]
        ],
    )
    df = build(tmp_path)
    assert sorted(df.pid) == ["123456-0010", "123456-0020"]
    assert df.set_index("pid").loc["123456-0010", "price"] == 800000  # the earlier sale, not the 30-day flip


def synthetic_sales(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 900, n)), unit="D")
    sub = rng.choice(["A-1", "A-2", "B-1"], n)
    sqft = rng.uniform(900, 3500, n)
    level = {"A-1": 500, "A-2": 350, "B-1": 250}
    ppsf = np.array([level[s] for s in sub]) * (1 + 0.0002 * np.arange(n)) * np.exp(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "pid": [f"{i:06d}-0001" for i in range(n)],
            "date": dates,
            "price": sqft * ppsf,
            "living_sqft": sqft,
            "lot_sqft": rng.uniform(3000, 9000, n),
            "subarea": sub,
            "zip": np.where(np.char.startswith(sub, "A"), "98103", "98118"),
            "grade": rng.integers(6, 10, n),
            "condition": 3,
            "baths": 2.0,
            "bedrooms": 3,
            "stories": 1.0,
            "basement_sqft": 0,
            "garage_sqft": 0,
            "age": 40,
            "renovated": 0,
            "views": 0,
            "waterfront": 0,
            "traffic_noise": 0,
            "year_built": 1980,
            "year_renovated": 0,
            "address": [f"{i} TEST ST" for i in range(n)],
            "area": "1",
        }
    )


def test_comps_and_market_features_never_look_ahead():
    df = synthetic_sales()
    comps = M.prior_comps(df, "subarea")
    # recompute one sale's comps by hand from strictly earlier sales in its subarea
    i = 2000
    earlier = df[(df.subarea == df.subarea[i]) & (df.date < df.date[i])].tail(M.COMPS)
    assert comps[i] == pytest.approx(np.median(np.log(earlier.price / earlier.living_sqft)))
    # changing a later sale's price cannot change an earlier sale's features
    later = df.copy()
    later.loc[i + 1 :, "price"] *= 10
    assert np.allclose(M.prior_comps(later, "subarea")[: i + 1], comps[: i + 1], equal_nan=True)
    idx = M.market_index(df)
    assert np.allclose(
        M.lagged_market(later.date[: i + 1], M.market_index(later))[: i // 2],
        M.lagged_market(df.date[: i + 1], idx)[: i // 2],
        equal_nan=True,  # the first month has no earlier months on either side
    )


def test_conformal_ranges_cover_at_the_stated_rate_and_widen_where_noisier():
    rng = np.random.default_rng(1)
    pred = rng.uniform(12, 15, 40000)
    noise = np.where(pred > 14, 0.3, 0.1)  # expensive homes are noisier
    y = pred + rng.normal(0, 1, len(pred)) * noise
    cal, test = slice(0, 20000), slice(20000, None)
    for conf in (M.Conformal.fit(y[cal], pred[cal]), M.MondrianConformal.fit(y[cal], pred[cal], bands=4)):
        lo, hi = conf.interval(pred[test], 0.8)
        assert abs(((np.exp(y[test]) >= lo) & (np.exp(y[test]) <= hi)).mean() - 0.8) < 0.015
    mc = M.MondrianConformal.fit(y[cal], pred[cal], bands=4)
    assert mc.q[0.8][-1] > 2 * mc.q[0.8][0]
    top = pred[test] > 14.25
    lo, hi = mc.interval(pred[test], 0.8)
    assert abs(((np.exp(y[test]) >= lo) & (np.exp(y[test]) <= hi))[top].mean() - 0.8) < 0.03


def test_error_summary():
    s = M.summary(np.array([100.0, 100, 100, 100]), np.array([104.0, 92, 130, 100]))
    assert s == {"median_abs_pct_error": 6.0, "within_5pct": 50.0, "within_10pct": 75.0, "within_20pct": 75.0}


@pytest.fixture
def client(monkeypatch):
    sales = synthetic_sales()
    f = M.features(sales)
    X = M.NUMERIC + M.CATEGORICAL
    model = M.hedonic().fit(f[X], np.log(f.price))
    conf = M.MondrianConformal.fit(np.log(f.price.to_numpy()), model.predict(f[X]))
    homes = sales.drop(columns=["date", "price", "age", "renovated"]).head(50)
    bundle = {"model": model, "conformal": conf, "trained_through": "2024-06-01"}
    monkeypatch.setattr(
        webapp, "state", lambda: (bundle, sales, M.market_index(sales), homes, homes.address.str.upper())
    )
    return TestClient(webapp.app)


def test_api_search_estimate_and_validation(client):
    hits = client.get("/api/search", params={"q": "12 test"}).json()
    assert hits and all("12" in h["address"] for h in hits)
    r = client.get(f"/api/estimate/{hits[0]['pid']}")
    assert r.status_code == 200
    d = r.json()
    assert d["range_80"][0] < d["estimate"] < d["range_80"][1]
    assert all(c["living_sqft"] > 0 for c in d["comparable_sales"])
    assert client.get("/api/estimate/not-a-parcel").status_code == 400
    assert client.get("/api/estimate/999999-9999").status_code == 404
    assert client.get("/api/search", params={"q": "ab"}).status_code == 422
    assert client.get("/").status_code == 200
