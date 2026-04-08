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
            RV_train, RV_val, RV_test    -> np.ndarray (raw, no scaling)
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
    val_EAD = X_val['BILL_AMT1']
    test_EAD = X_test['BILL_AMT1']

    # Get splitted Revolving Balance (RV)
    train_RV = (X_train['BILL_AMT1'] - X_train['PAY_AMT1']).clip(lower=0)
    val_RV = (X_val['BILL_AMT1'] - X_val['PAY_AMT1']).clip(lower=0)
    test_RV = (X_test['BILL_AMT1'] - X_test['PAY_AMT1']).clip(lower=0)

    # 2. Scale — fitted only on train, transformed val and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 3. PCA — fitted only on train, transformed val and test
    pca_full = PCA(n_components=None, random_state=random_state)
    pca_full.fit(X_train_scaled)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(
        cumulative_variance, variance_threshold) + 1)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    actual_variance = float(np.sum(pca.explained_variance_ratio_))

    print(
        f"\nPCA: {X_train.shape[1]} features → {n_components} components "
        f"({actual_variance:.2%} variance explained)"
    )
    print(
        f"Sizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}"
    )

    return {
        "X_train":   X_train_pca,
        "X_val":     X_val_pca,
        "X_test":    X_test_pca,
        "y_train":   y_train.values,
        "y_val":     y_val.values,
        "y_test":    y_test.values,
        "EAD_train": train_EAD.values,
        "EAD_val":   val_EAD.values,
        "EAD_test":  test_EAD.values,
        "RV_train":  train_RV.values,
        "RV_val":    val_RV.values,
        "RV_test":   test_RV.values,
        "r_train":   r_train,
        "r_val":     r_val,
        "r_test":    r_test,
    }


def val_eval_cal_split(X_val: pd.DataFrame, y_val: pd.Series, r_val: pd.Series, EAD_val: pd.Series, RV_val: pd.Series, val_size: float = 0.3, random_state: int = 42) -> dict:
    """
    Split the validation set into a smaller validation set for model evaluation and a calibration set for probability calibration.

    Parameters:
        X_val : pd.DataFrame - Validation features.
        y_val : pd.Series - Validation target variable.
        r_val : pd.Series - Validation interest rates.
        EAD_val : pd.Series - Validation exposure at default.
        RV_val : pd.Series - Validation revolving balance.
        val_size : float - Proportion of the original validation set to use for model evaluation (default 0.5).
        random_state : int - Random seed for reproducibility.

    Returns:
        dict with keys:
            X_val_eval, y_val_eval, r_val_eval, EAD_val_eval, RV_val_eval   -> subsets for model evaluation
            X_val_cal, y_val_cal, r_val_cal, EAD_val_cal, RV_val_cal -> subsets for calibration
    """
    eval_ratio_adjusted = val_size

    X_val_eval, X_val_cal, y_val_eval, y_val_cal, r_val_eval, r_val_cal, EAD_val_eval, EAD_val_cal, RV_val_eval, RV_val_cal = train_test_split(
        X_val, y_val, r_val, EAD_val, RV_val,
        test_size=eval_ratio_adjusted,
        stratify=y_val,
        random_state=random_state,
    )

    print(
        f"\nValidation split — Eval: {len(y_val_eval)}, Calib: {len(y_val_cal)}"
    )

    return {
        "X_val_eval": X_val_eval,
        "y_val_eval": y_val_eval,
        "r_val_eval": r_val_eval,
        "EAD_val_eval": EAD_val_eval,
        "RV_val_eval": RV_val_eval,
        "X_val_cal": X_val_cal,
        "y_val_cal": y_val_cal,
        "r_val_cal": r_val_cal,
        "EAD_val_cal": EAD_val_cal,
        "RV_val_cal": RV_val_cal,
    }
