import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def split_scale_pca(
    X: pd.DataFrame,
    y: pd.Series,
    r: pd.Series,
    test_size: float = 0.05,
    val_size: float = 0.25,
    variance_threshold: float = 0.90,
    random_state: int = 42,
) -> dict:
    """
    Split data into train/validation/test, scale features, and apply PCA.
    r is split alongside X and y but is not scaled or PCA-transformed.

    Parameters:
        X : pd.DataFrame - Feature matrix.
        y : pd.Series - Target vector.
        r : pd.Series - Client-level interest rate.
        test_size : float - Proportion of the full dataset reserved for test.
        val_size : float - Proportion of the full dataset reserved for validation.
        variance_threshold : float - Minimum cumulative explained variance for PCA (default 0.90).
        random_state : int

    Returns:
        dict with keys:
            X_train, X_val, X_test       -> np.ndarray (PCA-transformed)
            y_train, y_val, y_test       -> np.ndarray
            r_train, r_val, r_test       -> pd.Series (raw, no scaling)
            EAD_train, EAD_val, EAD_test -> np.ndarray (raw, no scaling)
            scaler                       -> fitted StandardScaler
            pca                          -> fitted PCA object
            n_components                 -> int, number of components kept
            explained_variance           -> float, cumulative variance explained
    """

    # 1. Split  (train | val | test)
    val_ratio_adjusted = val_size / (1.0 - test_size)

    # Pass r alongside X and y to guarantee alignment
    X_temp, X_test, y_temp, y_test, r_temp, r_test = train_test_split(
        X, y, r,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val, r_train, r_val = train_test_split(
        X_temp, y_temp, r_temp,
        test_size=val_ratio_adjusted,
        stratify=y_temp,
        random_state=random_state,
    )
    
    # Get splited EAD
    train_EAD = X_train['BILL_AMT1']
    val_EAD   = X_val['BILL_AMT1']
    test_EAD  = X_test['BILL_AMT1']

    # 2. Scale — fitted only on train, transformed val and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # 3. PCA — fitted only on train, transformed val and test
    pca_full = PCA(n_components=None, random_state=random_state)
    pca_full.fit(X_train_scaled)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    actual_variance = float(np.sum(pca.explained_variance_ratio_))

    print(
        f"\nPCA: {X_train.shape[1]} features → {n_components} components "
        f"({actual_variance:.2%} variance explained)"
    )
    print(
        f"Sizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}"
    )

    return {
        "X_train": X_train_pca,
        "X_val":   X_val_pca,
        "X_test":  X_test_pca,
        "y_train": y_train.values,
        "y_val":   y_val.values,
        "y_test":  y_test.values,
        "EAD_train": train_EAD.values,
        "EAD_val":   val_EAD.values,
        "EAD_test":  test_EAD.values,
        "r_train": r_train,
        "r_val":   r_val,
        "r_test":  r_test,
        "scaler":  scaler,
        "pca":     pca,
        "n_components":       n_components,
        "explained_variance": actual_variance,
    }
