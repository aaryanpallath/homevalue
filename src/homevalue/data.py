"""King County Assessor extracts -> one clean table of arm's-length single-family sales.

A recorded sale is kept only if it is a market transaction for an existing house:
  principal use residential (6); property type 'land with previously used building' (3) or 'household, single
  family' (11); a warranty deed (2) or statutory warranty deed (3) -- the county switched most sales from code 3 to
  code 2 in 2020, so filtering on 3 alone keeps 2019 and drops nearly every later sale; sale reason 'none' (1),
  which excludes foreclosures, divorces, gifts
  and estate transfers; no sale warning codes; one parcel per transaction; price over $50,000.
Each sale is joined to the parcel's one residential building and to the parcel record.

Caveat: building attributes are the assessor's current record, not the state at sale time, so a house remodeled
after it sold looks like it sold remodeled."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

BASE = "https://aqua.kingcounty.gov/extranet/assessor/"
FILES = {
    "EXTR_RPSale.csv": "Real%20Property%20Sales.zip",
    "EXTR_ResBldg.csv": "Residential%20Building.zip",
    "EXTR_Parcel.csv": "Parcel.zip",
}
VIEWS = [
    "MtRainier",
    "Olympics",
    "Cascades",
    "Territorial",
    "SeattleSkyline",
    "PugetSound",
    "LakeWashington",
    "LakeSammamish",
    "SmallLakeRiverCreek",
    "OtherView",
]


def fetch(data_dir: Path) -> None:
    for csv_name, zip_name in FILES.items():
        if not (data_dir / csv_name).exists():
            z = data_dir / zip_name.replace("%20", "_")
            urllib.request.urlretrieve(BASE + zip_name, z)
            zipfile.ZipFile(z).extractall(data_dir)


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    opts = pacsv.ConvertOptions(
        include_columns=cols, strings_can_be_null=True, column_types={c: "string" for c in cols}
    )
    return pacsv.read_csv(path, read_options=pacsv.ReadOptions(encoding="latin-1"), convert_options=opts).to_pandas()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.strip(), errors="coerce")


def parcel_id(major: pd.Series, minor: pd.Series) -> pd.Series:
    return major.str.strip().str.zfill(6) + "-" + minor.str.strip().str.zfill(4)


def build(data_dir: str | Path, start: str = "2019-01-01", end: str = "2026-10-01") -> pd.DataFrame:
    data_dir = Path(data_dir)
    fetch(data_dir)
    s = _read(
        data_dir / "EXTR_RPSale.csv",
        [
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
        ],
    )
    s["date"] = pd.to_datetime(s.DocumentDate.str.strip(), format="%m/%d/%Y", errors="coerce")
    s["price"] = _num(s.SalePrice)
    keep = (
        (_num(s.PrincipalUse) == 6)
        & _num(s.PropertyType).isin([3, 11])
        & _num(s.SaleInstrument).isin([2, 3])
        & (_num(s.SaleReason) == 1)
        & (s.SaleWarning.fillna("").str.strip() == "")
        & (s.price > 50_000)
        & (s.date >= start)
        & (s.date < end)
    )
    s = s[keep]
    s = s[~s.ExciseTaxNbr.duplicated(keep=False)]  # one parcel per transaction
    s = s.assign(pid=parcel_id(s.Major, s.Minor))[["pid", "date", "price"]]

    b = _read(
        data_dir / "EXTR_ResBldg.csv",
        [
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
        ],
    )
    b["pid"] = parcel_id(b.Major, b.Minor)
    b = b[~b.pid.duplicated(keep=False)]  # exactly one residential building on the parcel
    b = b[_num(b.NbrLivingUnits) == 1]
    num = [
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
    for c in num:
        b[c] = _num(b[c])
    b["baths"] = b.BathFullCount + 0.75 * b.Bath3qtrCount + 0.5 * b.BathHalfCount
    b["zip"] = b.ZipCode.str.strip().str[:5]
    b["address"] = b.Address.str.strip().str.replace(r"\s+", " ", regex=True).str.replace(r" \d{5}$", "", regex=True)

    p = _read(
        data_dir / "EXTR_Parcel.csv",
        ["Major", "Minor", "Area", "SubArea", "SqFtLot", "WfntLocation", "TrafficNoise", *VIEWS],
    )
    p["pid"] = parcel_id(p.Major, p.Minor)
    p["area"] = p.Area.str.strip()
    p["subarea"] = p.area + "-" + p.SubArea.str.strip()
    p["lot_sqft"] = _num(p.SqFtLot)
    p["views"] = sum(_num(p[v]).fillna(0) for v in VIEWS)
    p["waterfront"] = (_num(p.WfntLocation).fillna(0) > 0).astype(int)
    p["traffic_noise"] = _num(p.TrafficNoise).fillna(0)

    df = s.merge(
        b[
            [
                "pid",
                "address",
                "zip",
                "Stories",
                "BldgGrade",
                "SqFtTotLiving",
                "SqFtTotBasement",
                "SqFtGarageAttached",
                "Bedrooms",
                "baths",
                "YrBuilt",
                "YrRenovated",
                "Condition",
            ]
        ],
        on="pid",
    ).merge(p[["pid", "area", "subarea", "lot_sqft", "views", "waterfront", "traffic_noise"]], on="pid")
    df = df.rename(
        columns={
            "Stories": "stories",
            "BldgGrade": "grade",
            "SqFtTotLiving": "living_sqft",
            "SqFtTotBasement": "basement_sqft",
            "SqFtGarageAttached": "garage_sqft",
            "Bedrooms": "bedrooms",
            "YrBuilt": "year_built",
            "YrRenovated": "year_renovated",
            "Condition": "condition",
        }
    )
    df = df[(df.living_sqft >= 300) & (df.living_sqft <= 15_000) & (df.lot_sqft > 0) & df.zip.str.match(r"^98\d{3}$")]
    # sold before it was built, or re-sold within 90 days (flips and corrections) are dropped
    df = df[df.year_built <= df.date.dt.year + 1].sort_values(["pid", "date"])
    gap = df.groupby("pid").date.diff().dt.days
    df = df[~(gap < 90)]
    # a few recorded prices are wildly off for the house (typos, partial interests): trim extreme $/sqft
    ppsf = df.price / df.living_sqft
    if len(df) >= 1000:  # percentile trimming only means something on a large sample
        lo, hi = ppsf.quantile([0.002, 0.998])
        df = df[(ppsf >= lo) & (ppsf <= hi)]
    df["age"] = df.date.dt.year - df.year_built
    df["renovated"] = (df.year_renovated.fillna(0) > 0).astype(int)
    return df.sort_values("date").reset_index(drop=True)


def load(data_dir: str | Path) -> pd.DataFrame:
    cache = Path(data_dir) / "sales.parquet"
    if not cache.exists():
        build(data_dir).to_parquet(cache)
    return pd.read_parquet(cache)


__all__ = ["build", "load", "np"]


def homes(data_dir: str | Path) -> pd.DataFrame:
    """Every single-family parcel with one house on it (sold recently or not), for looking up an estimate."""
    data_dir = Path(data_dir)
    cache = data_dir / "homes.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    fetch(data_dir)
    b = _read(
        data_dir / "EXTR_ResBldg.csv",
        [
            "Major",
            "Minor",
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
        ],
    )
    b["pid"] = parcel_id(b.Major, b.Minor)
    b = b[~b.pid.duplicated(keep=False) & (_num(b.NbrLivingUnits) == 1)]
    for c in [
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
    ]:
        b[c] = _num(b[c])
    p = _read(
        data_dir / "EXTR_Parcel.csv",
        ["Major", "Minor", "Area", "SubArea", "SqFtLot", "WfntLocation", "TrafficNoise", *VIEWS],
    )
    p["pid"] = parcel_id(p.Major, p.Minor)
    h = b.merge(p, on="pid")
    out = pd.DataFrame(
        {
            "pid": h.pid,
            "address": h.Address.str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r" \d{5}$", "", regex=True),
            "zip": h.ZipCode.str.strip().str[:5],
            "stories": h.Stories,
            "grade": h.BldgGrade,
            "living_sqft": h.SqFtTotLiving,
            "basement_sqft": h.SqFtTotBasement,
            "garage_sqft": h.SqFtGarageAttached,
            "bedrooms": h.Bedrooms,
            "baths": h.BathFullCount + 0.75 * h.Bath3qtrCount + 0.5 * h.BathHalfCount,
            "year_built": h.YrBuilt,
            "year_renovated": h.YrRenovated,
            "condition": h.Condition,
            "area": h.Area.str.strip(),
            "subarea": h.Area.str.strip() + "-" + h.SubArea.str.strip(),
            "lot_sqft": _num(h.SqFtLot),
            "views": sum(_num(h[v]).fillna(0) for v in VIEWS),
            "waterfront": (_num(h.WfntLocation).fillna(0) > 0).astype(int),
            "traffic_noise": _num(h.TrafficNoise).fillna(0),
        }
    )
    out = out[
        (out.living_sqft >= 300)
        & (out.lot_sqft > 0)
        & out.zip.str.match(r"^98\d{3}$", na=False)
        & out.address.str.len().gt(3)
    ]
    out = out.reset_index(drop=True)
    out.to_parquet(cache)
    return out
