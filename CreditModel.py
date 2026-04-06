import numpy as np
import pandas as pd

LGD = 0.75


def compute_thresholds(r: pd.Series, LGD: float = LGD) -> pd.Series:
    """
    Compute the break-even PD threshold for each client.

    Parameters:
        r : np.ndarray or pd.Series, shape (n_clients,) - Interest Rate for each client.
        LGD : float - Loss Given Default (assumed constant for the industry).

    Returns:
        tresholds : np.ndarray, shape (n_clients,) - Break-even PD threshold for each client.
    """
    thresholds = r / (r + LGD)

    return thresholds


def get_predictions(model, X: pd.DataFrame, r: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Get predicted probabilities and classifications based on the model and thresholds.
    
    Parameters:
        model: fitted model with a predict_proba method
        X: pandas DataFrame of features
        r: pandas Series of interest rate for each client
    
    Returns:
        probabilities: numpy array of predicted probabilities for the positive class
        predictions: numpy array of binary classifications based on the computed thresholds
    """
    # Get probabilities for the positive class
    probabilities = model.predict_proba(X)[:, 1]  
    
    # Classification thresholds
    thresholds = compute_thresholds(r)
    
    # Classify based on threshold - Lend if PD_i < threshold_i, i.e. expected PnL > 0
    predictions = (probabilities <= thresholds).astype(int)
    
    return probabilities, predictions, thresholds


def compute_expected_pnl(PD: np.ndarray, r: pd.Series, EAD: pd.Series, LGD: float = LGD) -> np.ndarray:
    """
    Compute expected PnL per client.

    PnL = (1 - PD) * EAD * r  -  PD * EAD * LGD

    Parameters:
        PD  : predicted default probabilities, shape (n_clients,)
        r   : client-level interest rate,      shape (n_clients,)
        EAD : exposure at default (BILL_AMT1), shape (n_clients,)
        LGD : loss given default (constant),   default 0.75

    Returns:
        pnl : np.ndarray, shape (n_clients,)
    """
    expected_revenue = (1 - PD) * EAD * r
    expected_loss    =      PD  * EAD * LGD

    return expected_revenue - expected_loss


def realized_pnl(r: pd.Series, EAD: pd.Series, pred: np.ndarray, y: pd.Series, LGD: float = LGD) -> np.ndarray:
    """
    Compute realized PnL per client.

    Parameters:
        r           : client-level annual interest rate, shape (n_clients,)
        EAD         : exposure at default (BILL_AMT1),   shape (n_clients,)
        predictions : lending decision (1=lend, 0=reject), shape (n_clients,)
        y_true      : actual default outcome (1=default, 0=paid), shape (n_clients,)
        LGD         : loss given default, default 0.75

    Returns:
        pnl : np.ndarray, shape (n_clients,)
    """
    pnl = np.where(pred == 0, 0,           # didn't lend
          np.where(y   == 0,  EAD * r,     # lent → no default
                              -EAD * LGD)) # lent → default

    return pnl
