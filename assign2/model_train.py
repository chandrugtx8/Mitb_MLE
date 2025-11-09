# model_train.py
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from typing import Dict, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)

RANDOM_STATE = 42


# ---------- loaders ----------
def load_feature_store() -> pd.DataFrame:
    folder = "datamart/gold/feature_store"
    return ds.dataset(folder, format="parquet").to_table().to_pandas()


def load_label_store() -> pd.DataFrame:
    folder = "datamart/gold/label_store"
    return ds.dataset(folder, format="parquet").to_table().to_pandas()


# ---------- build training frame ----------
def build_training_df() -> pd.DataFrame:
    df_feat = load_feature_store()
    df_lbl = load_label_store()[["Customer_ID", "label"]]  # only what we need
    df = df_feat.merge(df_lbl, on="Customer_ID", how="inner")

    # Ensure date column exists and is datetime (for time-based splits)
    if "feature_snapshot_date" in df.columns:
        df["feature_snapshot_date"] = pd.to_datetime(df["feature_snapshot_date"])
    else:
        raise RuntimeError(
            "feature_snapshot_date not found in feature store; required for OOT split."
        )

    print(f"✅ Merged features+labels: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


# ---------- feature prep ----------
def get_feature_columns(df_train: pd.DataFrame) -> List[str]:
    """Pick numeric/bool columns from TRAIN only and return list."""
    # drop obvious non-features if present
    drop_cols = {"label", "Customer_ID", "feature_snapshot_date"}
    X = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors="ignore")
    cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    return cols


def prepare_X(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Select feature columns and clean types/NaNs consistently."""
    X = df.reindex(columns=feature_cols).copy()
    # cast bool->int
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    # handle inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def split_time_slices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return dict of time-based splits: train, val, test, oot (by month)."""
    months = (
        df["feature_snapshot_date"]
        .dt.to_period("M")
        .sort_values()
        .unique()
        .tolist()
    )
    if len(months) < 2:
        raise RuntimeError("Not enough months to create OOT/Test splits.")

    # OOT = most recent month
    oot_m = months[-1]
    # TEST = 2nd most recent (if exists)
    test_m = months[-2] if len(months) >= 2 else None
    # VAL = 3rd most recent (if exists)
    val_m = months[-3] if len(months) >= 3 else None

    df_oot = df[df["feature_snapshot_date"].dt.to_period("M") == oot_m]
    df_test = df[df["feature_snapshot_date"].dt.to_period("M") == test_m] if test_m else pd.DataFrame()
    df_val = df[df["feature_snapshot_date"].dt.to_period("M") == val_m] if val_m else pd.DataFrame()

    # TRAIN = all months before VAL (if VAL exists), else before TEST, else before OOT
    if val_m:
        cutoff = val_m
    elif test_m:
        cutoff = test_m
    else:
        cutoff = oot_m

    df_train = df[df["feature_snapshot_date"].dt.to_period("M") < cutoff]

    # if train is empty (small dataset), fallback to random split but keep OOT reserved
    if df_train.empty:
        print("⚠️ TRAIN empty with time split; falling back to random split (keep OOT).")
        df_non_oot = df[df["feature_snapshot_date"].dt.to_period("M") != oot_m]
        if df_non_oot.empty:
            raise RuntimeError("All rows are in OOT month; cannot train.")
        df_train, df_val = train_test_split(
            df_non_oot, test_size=0.2, random_state=RANDOM_STATE, stratify=df_non_oot["label"]
        )
        df_test = pd.DataFrame()  # no clean TEST in this fallback path

    return {"train": df_train, "val": df_val, "test": df_test, "oot": df_oot}


# ---------- evaluation ----------
def evaluate(model, X: pd.DataFrame, y: pd.Series, label: str) -> Dict[str, float]:
    out = {"split": label}

    if X.empty or y.empty:
        out.update({"auc": np.nan, "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan})
        return out

    # probabilities if available (for AUC)
    try:
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = np.nan

    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)

    out.update({"auc": float(auc), "accuracy": float(acc), "precision": float(pr), "recall": float(rc), "f1": float(f1)})
    return out


def print_report(y_true, y_pred, title: str):
    print(f"\n📊 Classification report ({title}):\n")
    print(classification_report(y_true, y_pred, digits=3))


# ---------- main train/select/save ----------
def train_and_select(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
):
    """Train 2 models and select best on validation AUC (tie-breaker F1)."""

    candidates = []

    # 1) RandomForest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_val = evaluate(rf, X_val, y_val, "val")
    candidates.append(("RandomForest", rf, rf_val))

    # 2) LogisticRegression
    lr = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        random_state=RANDOM_STATE,
        n_jobs=None,
    )
    lr.fit(X_train, y_train)
    lr_val = evaluate(lr, X_val, y_val, "val")
    candidates.append(("LogisticRegression", lr, lr_val))

    # Select best: higher AUC, then higher F1
    def sort_key(item):
        name, model, metrics = item
        return (np.nan_to_num(metrics["auc"], nan=-1.0), np.nan_to_num(metrics["f1"], nan=-1.0))

    candidates.sort(key=sort_key, reverse=True)
    best_name, best_model, best_val_metrics = candidates[0]

    print("\n✅ Model selection (by VAL AUC then F1):")
    for name, _, m in candidates:
        print(f"  - {name:18s}  AUC={m['auc']:.4f}  F1={m['f1']:.4f}")

    print(f"\n🏆 Selected: {best_name} (VAL AUC={best_val_metrics['auc']:.4f}, F1={best_val_metrics['f1']:.4f})")
    return best_name, best_model, best_val_metrics


def save_artifacts(model, feature_cols: List[str], metrics: Dict[str, Dict[str, float]]):
    Path("model_store").mkdir(parents=True, exist_ok=True)
    model_path = "model_store/best_model.pkl"
    joblib.dump(model, model_path)

    meta = {
        "feature_columns": feature_cols,
        "metrics": metrics,
    }
    with open("model_store/best_model.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n💾 Saved model -> {model_path}")
    print("💾 Saved meta  -> model_store/best_model.meta.json")


# ---------- main ----------
if __name__ == "__main__":
    # 1) Load & build
    df = build_training_df()

    # 2) Time splits
    splits = split_time_slices(df)
    df_tr, df_val, df_te, df_oot = splits["train"], splits["val"], splits["test"], splits["oot"]

    print(
        f"\n🗂️ Splits (rows): "
        f"TRAIN={len(df_tr)}, VAL={len(df_val)}, TEST={len(df_te)}, OOT={len(df_oot)}"
    )

    # 3) Feature columns from TRAIN only
    feature_cols = get_feature_columns(df_tr)
    print(f"🧮 Using {len(feature_cols)} numeric features")

    # 4) Prepare X/y for all splits
    def xy(df_):
        X_ = prepare_X(df_, feature_cols)
        y_ = df_["label"].astype(int)
        return X_, y_

    X_tr, y_tr = xy(df_tr)
    X_val, y_val = xy(df_val) if not df_val.empty else (pd.DataFrame(columns=feature_cols), pd.Series(dtype=int))
    X_te, y_te = xy(df_te) if not df_te.empty else (pd.DataFrame(columns=feature_cols), pd.Series(dtype=int))
    X_oot, y_oot = xy(df_oot)

    # 5) Train & select best on validation
    if df_val.empty:
        # If no separate VAL, fall back to a train/val split inside train
        print("⚠️ No VAL month available; making a random 80/20 split inside TRAIN for model selection.")
        X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
        )
        model_name, model, _ = train_and_select(X_tr2, y_tr2, X_val2, y_val2)
        # Merge back all TRAIN for final training
        X_tr_final, y_tr_final = X_tr, y_tr
    else:
        model_name, model, _ = train_and_select(X_tr, y_tr, X_val, y_val)
        # Final training on TRAIN+VAL
        X_tr_final = pd.concat([X_tr, X_val], axis=0)
        y_tr_final = pd.concat([y_tr, y_val], axis=0)

    model.fit(X_tr_final, y_tr_final)

    # 6) Evaluate on TEST & OOT (and VAL if exists)
    metrics_all = {}
    if not df_val.empty:
        m_val = evaluate(model, X_val, y_val, "VAL")
        metrics_all["val"] = m_val
        print_report(y_val, model.predict(X_val), "VAL")

    if not df_te.empty:
        m_test = evaluate(model, X_te, y_te, "TEST")
        metrics_all["test"] = m_test
        print_report(y_te, model.predict(X_te), "TEST")

    m_oot = evaluate(model, X_oot, y_oot, "OOT")
    metrics_all["oot"] = m_oot
    print_report(y_oot, model.predict(X_oot), "OOT")

    # 7) Save artifacts
    save_artifacts(model, feature_cols, metrics_all)

    print("\n✅ Done.")
