import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


def interest_rate_creation(risk_premium: pd.Series, PD_i: pd.Series) -> pd.DataFrame:
    """
    Create a DataFrame for interest rates based on risk premium and fixed costs.

    Parameters:
        risk_premium: pandas Series of calculated risk premiums for each individual
        PD_i: pandas Series of predicted probabilities of default for each individual

    Returns:
        InterestRate: pandas DataFrame containing the components of the interest rate and total interest rate
    """
    BASE_RATE = 0.025
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
    df['BucketRate'] = np.array(
        [centroids[sorted_idx[label_map[l]]] for l in raw_labels])
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


def build_rates(PD: pd.Series, LGD: int = 0.75, n_buckets: int = 5) -> pd.DataFrame:
    """
    Helper function to build interest rates for each client based on PD and risk premium.
    
    Parameters:
        PD: pandas Series of predicted probabilities of default for each individual
        LGD: Loss Given Default (default 0.75)
        n_buckets: number of pricing buckets to discretize into (default 5)
    """
    df = interest_rate_creation(PD * LGD, PD)
    df = discretize_into_buckets(df, n_buckets=n_buckets)
    return df
