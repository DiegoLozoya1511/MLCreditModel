import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


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


def interest_rate_creation(risk_premium: pd.Series, PD_i: pd.Series) -> pd.DataFrame:
    """
    Create a DataFrame for interest rates based on risk premium and fixed costs.
    
    Parameters:
        risk_premium: pandas Series of calculated risk premiums for each individual
        
    Returns:
        InterestRate: pandas DataFrame containing the components of the interest rate and total interest rate
    """
    BASE_RATE = 0.035
    INFLATION_RATE = 0.0450
    LIQUIDITY_PREMIUM = 0.05
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

    InterestRate['TotalInterestRate'] = InterestRate.sum(
        axis=1)

    InterestRate['PD_i'] = PD_i
    
    return InterestRate


def discretize_into_buckets(interest_rate_df: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """
    Discretize continuous interest rates into pricing buckets using 1D k-means.
    Minimizes within-bucket variance of TotalInterestRate.

    Parameters:
        interest_rate_df: DataFrame output of interest_rate_creation()
        n_buckets: number of pricing buckets (default 5)

    Returns:
        df: original DataFrame with added columns:
            - 'Bucket': integer label (0 = lowest risk, K-1 = highest)
            - 'BucketRate': the representative (centroid) rate for that bucket
            - 'QuantizationLoss': |TotalInterestRate - BucketRate| per client
    """
    df = interest_rate_df.copy()
    rates = df['TotalInterestRate'].values.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_buckets, random_state=42, n_init=20)
    kmeans.fit(rates)

    # Sort cluster labels by centroid value so Bucket 0 = cheapest
    centroids = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centroids)
    label_map = {old: new for new, old in enumerate(sorted_idx)}

    raw_labels = kmeans.labels_
    df['Bucket'] = np.array([label_map[l] for l in raw_labels])
    df['BucketRate'] = np.array([centroids[sorted_idx[label_map[l]]] for l in raw_labels])
    df['QuantizationLoss'] = np.abs(df['TotalInterestRate'] - df['BucketRate'])

    return df


def bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize bucket composition and quantization loss.

    Parameters:
        df: DataFrame output of discretize_into_buckets()

    Returns:
        summary: DataFrame with one row per bucket showing:
            - BucketRate: representative rate charged
            - Count / Share: number and % of clients
            - AvgContinuousRate: mean of TotalInterestRate within bucket
            - AvgPD: mean PD within bucket
            - AvgQuantizationLoss: mean |r* - r_bucket|
            - MaxQuantizationLoss: worst-case deviation within bucket
    """
    summary = (
        df.groupby('Bucket')
        .agg(
            BucketRate=('BucketRate', 'first'),
            Count=('TotalInterestRate', 'count'),
            AvgContinuousRate=('TotalInterestRate', 'mean'),
            AvgPD=('PD_i', 'mean'),
            AvgQuantizationLoss=('QuantizationLoss', 'mean'),
            MaxQuantizationLoss=('QuantizationLoss', 'max'),
        )
        .reset_index()
    )
    summary['Share'] = summary['Count'] / summary['Count'].sum()
    return summary


def calculate_interest_rate():
    """Main function to calculate interest rates based on PD and risk premium."""
    # Read the dataset
    data = pd.read_csv('Data/UCI_Credit_Card.csv')

    # Split the dataset into features and target variable
    target = 'default.payment.next.month'
    y = data[target]

    X = data.copy().drop(columns=['ID', target])
    
    model = joblib.load('Model/xgb_model_calibrated.pkl')
    X['PD_i'] = model.predict_proba(X)[:, 1]

    # Calculate risk premium using the formula
    LGD = 0.75 
    risk_premium = X['PD_i'] * LGD

    # Create interest rate DataFrame
    InterestRate = interest_rate_creation(risk_premium, X['PD_i'])
    
    # Discretize into buckets
    InterestRate = discretize_into_buckets(InterestRate, n_buckets=5)
    summary = bucket_summary(InterestRate)

    # Write results to CSV
    InterestRate.to_csv('Data/InterestRate.csv', index=False)

    print('\nRisk Premium:')
    print("Average PD:", np.mean(X['PD_i'].mean()))
    print("Average Risk Premium:", np.mean(risk_premium))
    
    print("\nBucket Summary:")
    print(summary.to_string(index=False))
    print("\nAvg Quantization Loss:", InterestRate['QuantizationLoss'].mean())


if __name__ == "__main__":
    calculate_interest_rate()
