"""
Preprocessing Pipeline  (Section VI.A)
======================================

Unified five-step policy:
  1. Drop / pseudonymise identifiers
  2. Clean non-numeric tokens
  3. Impute (median) & scale (StandardScaler)
  4. Extract cyclical temporal features
  5. Handle class imbalance via weighting (not resampling)

Also: create_unified_dataset() zero-pads features to max dimensionality.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------ #
#  Per-domain preprocessing                                            #
# ------------------------------------------------------------------ #
def preprocess_domain(
    df: pd.DataFrame,
    label_col: str,
    domain: str,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
    """Preprocess a single sub-dataset.

    Returns (X, y, feature_names, fitted_scaler).
    """
    df = df.copy()

    # 1. Drop identifiers
    id_cols = [
        "Flow ID", "FlowID", "flow_id",
        "Src IP", "Dst IP", "Source IP", "Destination IP",
        "IncidentId", "AlertId", "Id", "OrgId",
    ]
    df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True)

    # 2. Temporal features (cyclical)
    ts_cols = ["Timestamp", "timestamp", "Flow Start", "StartTime", "CreatedTime"]
    for col in ts_cols:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() > 0:
            hour = parsed.dt.hour.fillna(0)
            dow = parsed.dt.dayofweek.fillna(0)
            df[f"{col}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df[f"{col}_hour_cos"] = np.cos(2 * np.pi * hour / 24)
            df[f"{col}_dow_sin"] = np.sin(2 * np.pi * dow / 7)
            df[f"{col}_dow_cos"] = np.cos(2 * np.pi * dow / 7)
            df[f"{col}_is_weekend"] = (dow >= 5).astype(int)
        df.drop(columns=[col], inplace=True)

    # 3. Extract labels
    y = _extract_labels(df, label_col, domain)
    if label_col in df.columns:
        df.drop(columns=[label_col], inplace=True)

    # 4. Encode remaining categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].nunique() < 50:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            df.drop(columns=[col], inplace=True)

    # 5. Convert to float
    feature_names = df.columns.tolist()
    X = df.values.astype(np.float64)

    # 6. Clean infinities → NaN → median impute
    X = np.where(np.isinf(X), np.nan, X)
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    nan_locs = np.where(np.isnan(X))
    X[nan_locs] = np.take(col_medians, nan_locs[1])

    # 7. Winsorise (0.1 – 99.9 percentile)
    for j in range(X.shape[1]):
        lo, hi = np.percentile(X[:, j], [0.1, 99.9])
        X[:, j] = np.clip(X[:, j], lo, hi)

    # 8. Standardise
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X.astype(np.float32), y, feature_names, scaler


# ------------------------------------------------------------------ #
#  Label extraction                                                    #
# ------------------------------------------------------------------ #
def _extract_labels(df: pd.DataFrame, label_col: str, domain: str) -> np.ndarray:
    if label_col not in df.columns:
        return np.zeros(len(df), dtype=np.int32)

    labels = df[label_col]
    if domain == "soc":
        if labels.dtype == object:
            return (labels == "TP").astype(np.int32).values
        return (labels > 0).astype(np.int32).values

    if labels.dtype == object:
        return (
            ~labels.isin(["Normal", "Benign", "BENIGN"])
        ).astype(np.int32).values

    return (labels != 0).astype(np.int32).values


# ------------------------------------------------------------------ #
#  Unified (cross-domain) dataset                                      #
# ------------------------------------------------------------------ #
def create_unified_dataset(
    preprocessed: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-pad all domains to same dimensionality and concatenate.

    Parameters
    ----------
    preprocessed : dict
        {name: (X, y, feature_names)} for each domain.

    Returns
    -------
    X_unified, y_unified, domain_ids
        domain_ids is an integer array identifying the source domain.
    """
    max_d = max(X.shape[1] for X, _, _ in preprocessed.values())

    Xs, ys, ids = [], [], []
    for i, (name, (X, y, _)) in enumerate(preprocessed.items()):
        if X.shape[1] < max_d:
            pad = np.zeros((X.shape[0], max_d - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
        Xs.append(X)
        ys.append(y)
        ids.append(np.full(len(y), i, dtype=np.int32))

    X_u = np.vstack(Xs)
    y_u = np.concatenate(ys)
    d_u = np.concatenate(ids)

    print(f"\nUnified dataset: {X_u.shape[0]:,} samples, {X_u.shape[1]} features")
    print(f"  Attack rate: {y_u.mean():.2%}")
    return X_u, y_u, d_u
