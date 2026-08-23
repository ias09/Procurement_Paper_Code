import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COMMON_COLUMNS = [
    "po_id", "supplier_id", "material_or_category", "order_date",
    "promised_delivery_date", "actual_delivery_date", "quantity",
    "unit_price", "total_spend", "order_priority_or_type",
]


def derive_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Equations 1-5 from the proposal. Applied identically to both datasets."""
    df = df.copy()
    df["promised_lead_time"] = (df["promised_delivery_date"] - df["order_date"]).dt.days
    df["actual_lead_time"] = (df["actual_delivery_date"] - df["order_date"]).dt.days
    df["delay"] = df["actual_lead_time"] - df["promised_lead_time"]
    df["delay_days"] = df["delay"].clip(lower=0)
    df["late"] = (df["delay_days"] > 0).astype(int)
    return df


def qa_report(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- QA report: {name} ---")
    print(f"Rows: {len(df)}")
    neg_plt = (df["promised_lead_time"] < 0).sum()
    neg_alt = (df["actual_lead_time"] < 0).sum()
    dup_po = df["po_id"].duplicated().sum()
    nulls = df[["order_date", "promised_delivery_date", "actual_delivery_date"]].isna().sum().sum()
    print(f"Negative promised lead time: {neg_plt}")
    print(f"Negative actual lead time:   {neg_alt}")
    print(f"Duplicate PO ids:            {dup_po}")
    print(f"Null critical dates:         {nulls}")
    print(f"Late rate:                   {df['late'].mean():.4f}")
    print(f"Date range:                  {df['order_date'].min()} -> {df['order_date'].max()}")


def load_synthetic() -> pd.DataFrame:
    raw = pd.read_excel(DATA_DIR / "synthetic_raw.xlsx")
    df = pd.DataFrame({
        "po_id": raw["order_id"],
        "supplier_id": raw["supplier_id"],
        "material_or_category": raw["material_category"],
        "order_date": pd.to_datetime(raw["order_date"]),
        "promised_delivery_date": pd.to_datetime(raw["promised_delivery_date"]),
        "actual_delivery_date": pd.to_datetime(raw["actual_delivery_date"]),
        "quantity": raw["quantity"],
        "unit_price": raw["unit_price"],
        "total_spend": raw["total_spend"],
        "order_priority_or_type": raw["order_priority"],
    })

    df["plant_id"] = raw["plant_id"]
    df["buyer_id"] = raw["buyer_id"]

    df = derive_targets(df)

    mismatch_delay = (df["delay_days"] != raw["delay_days"]).sum()
    mismatch_late = (df["late"] != raw["late"]).sum()
    print(f"[synthetic] recomputed vs provided delay_days mismatches: {mismatch_delay}")
    print(f"[synthetic] recomputed vs provided late mismatches:       {mismatch_late}")

    return df


def load_dataset1() -> pd.DataFrame:
    raw = pd.read_excel(DATA_DIR / "dataset1_raw.xlsx", sheet_name="Data")
    df = pd.DataFrame({
        "po_id": raw["PO Number"],
        "supplier_id": raw["Supplier ID"],
        "material_or_category": raw["Category"],
        "order_date": pd.to_datetime(raw["PO Date"], dayfirst=True),
        "promised_delivery_date": pd.to_datetime(raw["Requested Delivery"], dayfirst=True),
        "actual_delivery_date": pd.to_datetime(raw["Actual Delivery"], dayfirst=True),
        "quantity": raw["Quantity"],
        "unit_price": raw["Unit Price"],
        "total_spend": raw["Line Net"],
        "order_priority_or_type": raw["PO Type"],
    })

    df["supplier_tier"] = raw["Supplier Tier"]
    df["supplier_risk"] = raw["Supplier Risk"]
    df["contract_type"] = raw["Contract Type"]
    df["single_source_flag"] = raw["Single Source Flag"]
    df["preferred_supplier"] = raw["Preferred Supplier"]
    df["maverick_spend"] = raw["Maverick Spend"]
    df["supplier_esg_score"] = raw["Supplier ESG Score"]

    df = derive_targets(df)

    mismatch_delay_sign = (df["delay"] != raw["Days Late"]).sum()
    print(f"[dataset1] recomputed delay vs provided 'Days Late' mismatches: {mismatch_delay_sign}")
    provided_late = (raw["On Time Delivery"] == "No").astype(int)
    mismatch_late = (df["late"] != provided_late).sum()
    print(f"[dataset1] recomputed late vs provided 'On Time Delivery' mismatches: {mismatch_late}")

    return df


def main():
    synthetic = load_synthetic()
    qa_report(synthetic, "synthetic (dev/train)")
    synthetic.to_csv(DATA_DIR / "synthetic_clean.csv", index=False)

    ds1 = load_dataset1()
    qa_report(ds1, "dataset1 (external validation)")
    ds1.to_csv(DATA_DIR / "dataset1_clean.csv", index=False)

    print("\nSaved: data/synthetic_clean.csv, data/dataset1_clean.csv")


if __name__ == "__main__":
    main()
