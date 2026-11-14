"""M1 - Data preparation for the perishable-stock project (M5 Walmart dataset).

Reads the three raw M5 files from data/raw/, keeps a documented subset
(food items, most recent N days, top-selling items), and writes ONE tidy
table to data/processed/m5_subset.csv.

Run from the project root:
    python src/data_prep.py
"""
import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/m5_subset.csv")

CATEGORY = "FOODS"      # keep only food items
N_ITEMS = 40            # how many items to keep
WINDOW_DAYS = 730       # use only the most recent N days (about 2 years)
CANDIDATE_BUFFER = 20   # extra top sellers examined, in case some have gaps
MIN_ZERO_RUN = 14       # a run of this many zero-sales days in a row = "unavailable"

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def find_sales_file(raw_dir: Path) -> Path:
    for name in ("sales_train_evaluation.csv", "sales_train_validation.csv"):
        path = raw_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No sales_train_evaluation.csv or sales_train_validation.csv in {raw_dir}"
    )


def load_sales_wide(path: Path):
    """Read the wide sales file (one column per day) using small dtypes."""
    header = pd.read_csv(path, nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_")]
    dtypes = {c: "int16" for c in day_cols}
    for c in ["id"] + ID_COLS:
        dtypes[c] = "category"
    sales = pd.read_csv(path, dtype=dtypes)
    return sales, day_cols


def select_items(sales, day_cols, category, n_items, window_days):
    """Pick the top-selling items of one category using a stated, objective rule.

    Rule: among items sold in every store, rank by average daily units over
    the window and keep the top n_items.
    """
    window = day_cols[-window_days:]
    cat = sales[sales["cat_id"].astype(str) == category].copy()
    cat["mean_units"] = cat[window].mean(axis=1)

    n_stores = cat["store_id"].nunique()
    stores_per_item = cat.groupby("item_id", observed=True)["store_id"].nunique()
    in_all_stores = stores_per_item[stores_per_item == n_stores].index

    ranking = (
        cat[cat["item_id"].isin(in_all_stores)]
        .groupby("item_id", observed=True)["mean_units"]
        .mean()
        .sort_values(ascending=False)
    )
    return ranking.head(n_items).index.astype(str).tolist()


def to_long(sales, day_cols, items, window_days):
    """Keep the chosen items and the window, then reshape wide -> long."""
    window = day_cols[-window_days:]
    sub = sales[sales["item_id"].astype(str).isin(items)][ID_COLS + window].copy()
    sub[ID_COLS] = sub[ID_COLS].astype(str)
    return sub.melt(id_vars=ID_COLS, value_vars=window, var_name="d", value_name="units")


def add_calendar(long_df, raw_dir):
    cal = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    keep = ["d", "date", "wm_yr_wk", "wday", "month", "year",
            "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
    df = long_df.merge(cal[keep], on="d", how="left")

    # Collapse the three state SNAP columns into one flag for the row's own state
    df["snap"] = 0
    for st in ("CA", "TX", "WI"):
        mask = df["state_id"] == st
        df.loc[mask, "snap"] = df.loc[mask, f"snap_{st}"]
    df = df.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    df["event_name_1"] = df["event_name_1"].fillna("none")
    df["event_type_1"] = df["event_type_1"].fillna("none")
    return df


def add_prices(df, raw_dir, items):
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices["item_id"].isin(items)]
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # A missing price means the item was NOT on sale in that store that week,
    # so a zero in sales there is not real zero demand. Drop those rows.
    n_missing = df["sell_price"].isna().sum()
    df = df.dropna(subset=["sell_price"])
    return df, int(n_missing)


def mark_unavailable(df, min_len):
    """Flag days that sit inside a run of >= min_len consecutive zero-sales days.

    Long zero runs on items that normally sell are almost always stock-outs or
    periods when the item was not stocked, so they are NOT real zero demand.
    The rows are kept (so the calendar stays complete) but flagged, and the
    forecaster should exclude them from training.
    """
    def run_length(s):
        is_zero = (s == 0)
        run_id = (is_zero != is_zero.shift()).cumsum()
        return is_zero.groupby(run_id).transform("sum")

    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    run = df.groupby(["store_id", "item_id"])["units"].transform(run_length)
    df["unavailable"] = ((df["units"] == 0) & (run >= min_len)).astype(int)
    return df


def keep_complete_items(df, candidates, n_items, n_stores, window_days):
    """Keep only items on sale in EVERY store for EVERY day of the window.

    candidates is in ranking order (best seller first), so the first n_items
    complete ones are the top sellers that satisfy the rule.
    """
    rows_per_item = df.groupby("item_id").size()
    complete = set(rows_per_item[rows_per_item == n_stores * window_days].index)
    kept = [i for i in candidates if i in complete][:n_items]
    rejected = [i for i in candidates if i not in complete and i in set(rows_per_item.index)]
    if len(kept) < n_items:
        raise ValueError(
            f"Only {len(kept)} items have complete price history; "
            f"raise CANDIDATE_BUFFER or lower --items."
        )
    return df[df["item_id"].isin(kept)].copy(), kept, rejected


def main(n_items=N_ITEMS, window_days=WINDOW_DAYS, raw_dir=RAW_DIR, out_path=OUT_PATH,
         min_zero_run=MIN_ZERO_RUN):
    raw_dir, out_path = Path(raw_dir), Path(out_path)

    print("Loading sales ...")
    sales, day_cols = load_sales_wide(find_sales_file(raw_dir))
    n_stores = sales["store_id"].nunique()
    print(f"  raw sales table: {sales.shape[0]} series x {len(day_cols)} days")

    candidates = select_items(sales, day_cols, CATEGORY, n_items + CANDIDATE_BUFFER, window_days)
    print(f"  ranked {len(candidates)} candidate {CATEGORY} items (top sellers, last {window_days} days)")

    df = to_long(sales, day_cols, candidates, window_days)
    df = add_calendar(df, raw_dir)
    df, n_missing = add_prices(df, raw_dir, candidates)

    df, items, rejected = keep_complete_items(df, candidates, n_items, n_stores, window_days)
    print(f"  kept {len(items)} items with a full price history in all {n_stores} stores")
    print(f"  top sellers rejected for gaps in price history: {rejected if rejected else 'none'}")

    df = mark_unavailable(df, min_zero_run)
    print(f"  days flagged 'unavailable' (zero-sales runs of {min_zero_run}+ days): "
          f"{100 * df['unavailable'].mean():.1f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print("  rows:", len(df))
    print("  series (store x item):", df.groupby(["store_id", "item_id"]).ngroups)
    print("  date range:", df["date"].min().date(), "to", df["date"].max().date())
    print("  share of zero-sales rows: %.1f%%" % (100 * (df["units"] == 0).mean()))
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("  missing values per column:", "none" if missing.empty else "\n" + missing.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=N_ITEMS)
    parser.add_argument("--window", type=int, default=WINDOW_DAYS)
    parser.add_argument("--zero-run", type=int, default=MIN_ZERO_RUN)
    args = parser.parse_args()
    main(n_items=args.items, window_days=args.window, min_zero_run=args.zero_run)