import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


class RatePricer:
    """
    Builds client-level interest rates from predicted default probabilities
    and discretizes them into pricing buckets via 1D k-means clustering.

    The full interest rate is decomposed into fixed macro components and a
    client-specific risk premium derived from PD and LGD. Continuous rates
    are then quantized into a discrete set of bucket rates to reflect
    real-world pricing constraints.

    Attributes
    ----------
    rates_df   : pd.DataFrame with one row per client containing rate
                 components, TotalInterestRate, Bucket, BucketRate, and
                 QuantizationLoss. Populated after calling price().
    summary_df : pd.DataFrame with one row per bucket containing aggregate
                 statistics. Populated after calling summarize().
    """

    # Fixed macro rate components (annualized)
    _BASE_RATE = 0.025
    _INFLATION_RATE = 0.045
    _LIQUIDITY_PREMIUM = 0.050
    _ADMIN_COSTS = 0.070
    _OPERATING_COSTS = 0.050

    _REAL_RATE = _BASE_RATE - _INFLATION_RATE

    def __init__(self, LGD: float = 0.75, n_buckets: int = 5, random_state: int = 42):
        """
        Parameters
        ----------
        LGD          : Loss Given Default. Used to derive the risk premium
                       as PD * LGD.
        n_buckets    : Number of pricing buckets for k-means discretization.
        random_state : Random seed for k-means reproducibility.
        """
        self.LGD = LGD
        self.n_buckets = n_buckets
        self.random_state = random_state

        self.rates_df = None
        self.summary_df = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def price(self, PD: pd.Series) -> "RatePricer":
        """
        Compute continuous interest rates and discretize into buckets.

        Parameters
        ----------
        PD : Predicted default probabilities, one per client.

        Returns
        -------
        self : Enables method chaining.
        """
        risk_premium = PD * self.LGD
        rates = self._build_rate_dataframe(risk_premium, PD)
        self.rates_df = self._discretize(rates)
        return self

    def summarize(self) -> "RatePricer":
        """
        Compute bucket-level aggregate statistics and store in summary_df.

        Returns
        -------
        self : Enables method chaining.
        """
        self._require_rates()
        self.summary_df = (
            self.rates_df.groupby("Bucket")
            .agg(
                BucketRate=("BucketRate", "first"),
                Count=("TotalInterestRate", "count"),
                AvgContinuousRate=("TotalInterestRate", "mean"),
                AvgPD=("PD_i", "mean"),
                AvgQuantizationLoss=("QuantizationLoss", "mean"),
                MaxQuantizationLoss=("QuantizationLoss", "max"),
            )
            .reset_index()
        )
        self.summary_df["Share"] = (
            self.summary_df["Count"] / self.summary_df["Count"].sum()
        )
        return self

    @property
    def bucket_rates(self) -> pd.Series:
        """
        Return the BucketRate column from rates_df as a Series.
        Convenience accessor used by downstream classes (RatePricer → CreditPortfolio).
        """
        self._require_rates()
        return self.rates_df["BucketRate"]

    @staticmethod
    def combined_summary(pricers: list["RatePricer"]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate rates_df across multiple RatePricer instances and compute
        a unified bucket summary over the combined population.

        Intended for reporting purposes where train, val, and test rates need
        to be summarized jointly without re-fitting a new pricer.

        Parameters
        ----------
        pricers : List of RatePricer instances that have already called price().

        Returns
        -------
        total_rates : pd.DataFrame — concatenation of all rates_df.
        summary_df  : pd.DataFrame — bucket summary over total_rates.
        """
        for i, p in enumerate(pricers):
            if p.rates_df is None:
                raise RuntimeError(
                    f"Pricer at index {i} has no rates. Call price() on all pricers first."
                )

        total_rates = pd.concat(
            [p.rates_df for p in pricers],
            ignore_index=True,
        )

        summary_df = (
            total_rates.groupby("Bucket")
            .agg(
                BucketRate=("BucketRate", "first"),
                Count=("TotalInterestRate", "count"),
                AvgContinuousRate=("TotalInterestRate", "mean"),
                AvgPD=("PD_i", "mean"),
                AvgQuantizationLoss=("QuantizationLoss", "mean"),
                MaxQuantizationLoss=("QuantizationLoss", "max"),
            )
            .reset_index()
        )
        summary_df["Share"] = summary_df["Count"] / summary_df["Count"].sum()

        return total_rates, summary_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_rate_dataframe(
        self, risk_premium: pd.Series, PD: pd.Series
    ) -> pd.DataFrame:
        """Assemble the rate component DataFrame for each client."""
        df = pd.DataFrame({
            "Real Rate":         self._REAL_RATE,
            "Inflation Rate":    self._INFLATION_RATE,
            "Risk Premium":      risk_premium.values,
            "Liquidity Premium": self._LIQUIDITY_PREMIUM,
            "Admin Costs":       self._ADMIN_COSTS,
            "Operating Costs":   self._OPERATING_COSTS,
        })
        df["TotalInterestRate"] = df.sum(axis=1)
        df["PD_i"] = PD.values
        return df

    def _discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign each client to a pricing bucket using 1D k-means on
        TotalInterestRate. Bucket 0 is always the lowest-rate bucket.
        """
        rates = df["TotalInterestRate"].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_buckets,
                        random_state=self.random_state, n_init=20)
        kmeans.fit(rates)

        centroids = kmeans.cluster_centers_.flatten()
        sorted_idx = np.argsort(centroids)
        label_map = {old: new for new, old in enumerate(sorted_idx)}

        raw_labels = kmeans.labels_
        df["Bucket"] = np.array([label_map[l] for l in raw_labels])
        df["BucketRate"] = np.array(
            [centroids[sorted_idx[label_map[l]]] for l in raw_labels])
        df["QuantizationLoss"] = np.abs(
            df["TotalInterestRate"] - df["BucketRate"])

        return df

    def _require_rates(self) -> None:
        if self.rates_df is None:
            raise RuntimeError(
                "No rates computed. Call price(PD) first."
            )
