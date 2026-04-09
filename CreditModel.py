import numpy as np
import pandas as pd

LGD = 0.75


def compute_thresholds(r: pd.Series, RV: pd.Series, EAD: pd.Series, LGD: float = LGD) -> pd.Series:
    """
    Compute the break-even PD threshold for each client.

    Parameters:
        r   : np.ndarray or pd.Series, shape (n_clients,) - Interest Rate for each client.
        RV  : np.ndarray or pd.Series, shape (n_clients,) - Revolving Balance (BILL_AMT1 - PAY_AMT1) for each client.
        EAD : np.ndarray or pd.Series, shape (n_clients,) - Exposure at Default (BILL_AMT1) for each client.
        LGD : float - Loss Given Default (assumed constant for the industry).

    Returns:
        tresholds : np.ndarray, shape (n_clients,) - Break-even PD threshold for each client.
    """
    numerator = RV * r
    denominator = RV * r + EAD * LGD
    thresholds = np.divide(numerator, denominator, out=np.zeros_like(
        numerator, dtype=float), where=denominator != 0)
    return np.clip(thresholds, 0, 1)


def get_predictions(model, X: pd.DataFrame, r: pd.Series, RV: pd.Series, EAD: pd.Series, LGD: float = LGD) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predicted probabilities and classifications based on the model and thresholds.

    Parameters:
        model: fitted model with a predict_proba method
        X: pandas DataFrame of features
        r: pandas Series of interest rate for each client
        RV: pandas Series of revolving balance for each client
        EAD: pandas Series of exposure at default for each client
        LGD: float, loss given default (default 0.75)

    Returns:
        probabilities: numpy array of predicted probabilities for the positive class
        predictions: numpy array of binary classifications based on the computed thresholds
        thresholds: numpy array of break-even PD thresholds for each client
    """
    # Get probabilities for the positive class
    probabilities = model.predict_proba(X)[:, 1]

    # Classification thresholds
    thresholds = compute_thresholds(r, RV, EAD, LGD)

    # Classify based on threshold - Lend if PD_i < threshold_i, i.e. expected PnL > 0
    predictions = (probabilities <= thresholds).astype(int)

    return probabilities, predictions, thresholds


def realized_pnl(r: pd.Series, RV: pd.Series, EAD: pd.Series, pred: np.ndarray, y: pd.Series, LGD: float = LGD) -> np.ndarray:
    """
    Compute realized PnL per client.

    Parameters:
        r    : client-level annual interest rate, shape (n_clients,)
        RV   : revolving balance (BILL_AMT1 - PAY_AMT1), shape (n_clients,)
        EAD  : exposure at default (BILL_AMT1),   shape (n_clients,)
        pred : lending decision (1=lend, 0=reject), shape (n_clients,)
        y    : actual default outcome (1=default, 0=paid), shape (n_clients,)
        LGD  : loss given default, default 0.75

    Returns:
        pnl : np.ndarray, shape (n_clients,)
    """
    pnl = np.where(pred == 0, 0,            # didn't lend
                   np.where(y == 0, RV * r,  # lent → no default
                            -EAD * LGD))    # lent → default

    return pnl


def benchmark_pnl(r: pd.Series, RV: pd.Series, EAD: pd.Series, y: pd.Series) -> np.ndarray:
    """
    Compute realized PnL per client under a benchmark strategy of lending to everyone.

    Parameters:
        r    : client-level annual interest rate, shape (n_clients,)
        RV   : revolving balance (BILL_AMT1 - PAY_AMT1), shape (n_clients,)
        EAD  : exposure at default (BILL_AMT1),   shape (n_clients,)
        y    : actual default outcome (1=default, 0=paid), shape (n_clients,)

    Returns:
        pnl : np.ndarray, shape (n_clients,)
    """
    pred_all = np.ones(len(y), dtype=int)

    return realized_pnl(r, RV, EAD, pred_all, y)
