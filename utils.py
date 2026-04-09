import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.05,
    val_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """
    Split data into train/validation/test.

    Parameters:
        X            : pd.DataFrame - Feature matrix.
        y            : pd.Series - Target vector.
        test_size    : float - Proportion of the full dataset reserved for test.
        val_size     : float - Proportion of the full dataset reserved for validation.
        random_state : int - Random seed for reproducibility.

    Returns:
        dict with keys:
            X_train, X_val, X_test       -> np.ndarray
            y_train, y_val, y_test       -> np.ndarray
            EAD_train, EAD_val, EAD_test -> np.ndarray
            RV_train, RV_val, RV_test    -> np.ndarray
    """

    # Split  (train | val | test)
    val_ratio_adjusted = val_size / (1.0 - test_size)

    # Pass r alongside X and y to guarantee alignment
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
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

    print(
        f"\nSizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}"
    )

    return {
        "X_train":   X_train,
        "X_val":     X_val,
        "X_test":    X_test,
        "y_train":   y_train.values,
        "y_val":     y_val.values,
        "y_test":    y_test.values,
        "EAD_train": train_EAD.values,
        "EAD_val":   val_EAD.values,
        "EAD_test":  test_EAD.values,
        "RV_train":  train_RV.values,
        "RV_val":    val_RV.values,
        "RV_test":   test_RV.values,
    }


def val_eval_cal_split(X_val: pd.DataFrame, y_val: pd.Series, EAD_val: pd.Series, RV_val: pd.Series, val_size: float = 0.3, random_state: int = 42) -> dict:
    """
    Split the validation set into a smaller validation set for model evaluation and a calibration set for probability calibration.

    Parameters:
        X_val : pd.DataFrame - Validation features.
        y_val : pd.Series - Validation target variable.
        EAD_val : pd.Series - Validation exposure at default.
        RV_val : pd.Series - Validation revolving balance.
        val_size : float - Proportion of the original validation set to use for model evaluation (default 0.5).
        random_state : int - Random seed for reproducibility.

    Returns:
        dict with keys:
            X_val_eval, y_val_eval, EAD_val_eval, RV_val_eval   -> subsets for model evaluation
            X_val_cal, y_val_cal, EAD_val_cal, RV_val_cal -> subsets for calibration
    """
    eval_ratio_adjusted = val_size

    X_val_eval, X_val_cal, y_val_eval, y_val_cal, EAD_val_eval, EAD_val_cal, RV_val_eval, RV_val_cal = train_test_split(
        X_val, y_val, EAD_val, RV_val,
        test_size=eval_ratio_adjusted,
        stratify=y_val,
        random_state=random_state,
    )

    print(
        f"\nValidation split — Eval: {len(y_val_eval)}, Cal: {len(y_val_cal)}"
    )

    return {
        "X_val_eval": X_val_eval,
        "y_val_eval": y_val_eval,
        "EAD_val_eval": EAD_val_eval,
        "RV_val_eval": RV_val_eval,
        "X_val_cal": X_val_cal,
        "y_val_cal": y_val_cal,
        "EAD_val_cal": EAD_val_cal,
        "RV_val_cal": RV_val_cal,
    }
