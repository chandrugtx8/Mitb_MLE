# model_infer.py
from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

FEATURE_DIR = "datamart/gold/feature_store"
DEFAULT_MODEL = "model_store/best_model.pkl"
DEFAULT_META  = "model_store/best_model.meta.json"
DEFAULT_OUT   = "outputs/preds.csv"

def load_feature_store() -> pd.DataFrame:
    return ds.dataset(FEATURE_DIR, format="parquet").to_table().to_pandas()

def pick_month(df: pd.DataFrame, month: str | None) -> tuple[pd.DataFrame, str | None]:
    if "feature_snapshot_date" not in df.columns:
        return df, None
    if month:
        dfm = df[df["feature_snapshot_date"].astype(str) == month]
        if dfm.empty:
            raise SystemExit(f"No rows for feature_snapshot_date={month}")
        return dfm, month
    latest = df["feature_snapshot_date"].max()
    return df[df["feature_snapshot_date"] == latest], str(latest)

def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["Customer_ID", "feature_snapshot_date", "label"], errors="ignore")
    num_cols = X.select_dtypes(include=["number", "bool"]).columns
    X = X[num_cols].copy()
    for c in X.select_dtypes(include=["bool"]).columns:
        X[c] = X[c].astype(int)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def align_to_training(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    # add any missing columns with 0
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        for c in missing:
            X[c] = 0
    # drop extras and reorder
    X = X.loc[:, feature_names]
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--meta",  default=DEFAULT_META,
                    help="Meta JSON saved at training time (contains feature_names)")
    ap.add_argument("--out",   default=DEFAULT_OUT)
    ap.add_argument("--feature-month", help="YYYY-MM-01; defaults to latest month in feature_store")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold (default 0.5)")
    args = ap.parse_args()

    print("🔹 Loading feature store …")
    df = load_feature_store()
    dfm, month = pick_month(df, args.feature_month)
    print(f"🔹 Scoring rows: {dfm.shape[0]} (month={month or 'ALL/none'})")

    print("🔹 Preparing features …")
    X = prep_features(dfm)

    # Load training meta to guarantee same feature order
    feature_names = None
    try:
        with open(args.meta, "r") as f:
            meta = json.load(f)
        feature_names = meta.get("feature_names")
    except Exception:
        pass
    if feature_names:
        X = align_to_training(X, feature_names)

    print(f"🔹 Loading model: {args.model}")
    model = joblib.load(args.model)

    print("🔹 Predicting …")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        # map decision scores to [0,1] roughly via logistic; keeps compatibility
        z = model.decision_function(X)
        proba = 1 / (1 + np.exp(-z))
    else:
        # fallback to predict as hard label
        proba = model.predict(X).astype(float)

    pred = (proba >= args.threshold).astype(int)

    out = pd.DataFrame({
        "Customer_ID": dfm.get("Customer_ID", pd.Series(index=dfm.index, dtype="object")),
        "feature_snapshot_date": dfm.get("feature_snapshot_date", pd.NaT),
        "score": proba,
        "pred": pred,
    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"✅ Saved predictions -> {args.out}")
    print(out.head())

if __name__ == "__main__":
    main()
