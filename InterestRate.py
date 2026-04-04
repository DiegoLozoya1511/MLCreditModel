import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


def out_of_fold_predictions(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Generate out-of-fold predictions for PD estimation.
    
    Parameters:
        pipeline: sklearn Pipeline object containing the model and preprocessing steps
        X: pandas DataFrame of features
        y: pandas Series of target variable (binary)
    
    Returns:
        PD_oof: numpy array of out-of-fold predicted probabilities for the positive class
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    PD_oof = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        PD_oof[val_idx] = pipeline.predict_proba(X.iloc[val_idx])[:, 1]

    return PD_oof


def compute_risk_premium(PD_i: pd.Series, LGD: int) -> pd.Series:
    """
    Calculate risk premium using the formula: Risk Premium = (PD_i * LGD) / (1 - PD_i)
    
    Parameters:
        PD_i: pandas Series of predicted probabilities of default for each individual
        LGD: Loss Given Default (assumed constant for the industry)
        
    Returns:
        risk_premium: pandas Series of calculated risk premiums for each individual
    """
    return (PD_i * LGD) / (1 - PD_i.clip(upper=0.9999))


def interest_rate_creation(risk_premium: pd.Series) -> pd.DataFrame:
    """
    Create a DataFrame for interest rates based on risk premium and fixed costs.
    
    Parameters:
        risk_premium: pandas Series of calculated risk premiums for each individual
        
    Returns:
        InterestRate: pandas DataFrame containing the components of the interest rate and total interest rate
    """
    BASE_RATE = 0.0675
    INFLATION_RATE = 0.0450
    LIQUIDITY_PREMIUM = 0.02
    ADMIN_COSTS = 0.07
    OPERATING_COSTS = 0.05

    REAL_RATE = BASE_RATE - INFLATION_RATE

    InterestRate = pd.DataFrame({
        'Real Rate': REAL_RATE,
        'Inflation Rate': INFLATION_RATE,
        'Risk Premium': risk_premium,
        'Liquidity Premium': LIQUIDITY_PREMIUM,
        'Admin Costs': ADMIN_COSTS,
        'Operating Costs': OPERATING_COSTS,
    })

    # CAT ceiling based on Banxico official data (Ualá, cards > $15,000 limit)
    CAT_CAP = 1.39  # 139%

    # Implied annual rate ceiling (CAT - fixed costs approximation)
    # CAT ≈ r_i + fixed costs overhead, so r_i cap ≈ CAT_CAP - fixed costs
    FIXED_COSTS = INFLATION_RATE + LIQUIDITY_PREMIUM + ADMIN_COSTS + OPERATING_COSTS
    RATE_CAP = CAT_CAP - FIXED_COSTS

    InterestRate['TotalInterestRate'] = InterestRate.sum(
        axis=1).clip(upper=RATE_CAP)

    return InterestRate


def calculate_interest_rate():
    """Main function to calculate interest rates based on PD and risk premium."""
    # Read the dataset
    data = pd.read_csv('Data/UCI_Credit_Card.csv')

    # Split the dataset into features and target variable
    target = 'default.payment.next.month'
    y = data[target]

    X = data.copy().drop(columns=['ID', target])

    # Pipeline
    pipeline = Pipeline([
        # Scaling
        ('scaler', StandardScaler()),
        # Logistic Regression
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    # Calculate PD using Out-of-Fold predictions
    PD_oof = out_of_fold_predictions(pipeline, X, y)
    X['PD_i'] = PD_oof

    # Calculate risk premium using the formula
    LGD = 0.75 
    risk_premium = compute_risk_premium(X['PD_i'], LGD)

    # Create interest rate DataFrame
    InterestRate = interest_rate_creation(risk_premium)

    # Write results to CSV
    InterestRate.to_csv('Data/InterestRate.csv', index=False)

    print("Out-of-Fold Predicted Probabilities (PD):")
    print(X.head())
    print("\nAverage PD:", np.mean(X['PD_i'].mean()))
    print("\nAverage Risk Premium:", np.mean(risk_premium))
    print("\nInterest Rate DataFrame:")
    print(InterestRate.head())
    print("\nMax Total Interest Rate:",
          InterestRate['TotalInterestRate'].max())
    print("\nMin Total Interest Rate:",
          InterestRate['TotalInterestRate'].min())
    print("\nAverage Total Interest Rate:",
          InterestRate['TotalInterestRate'].mean())


if __name__ == "__main__":
    calculate_interest_rate()
