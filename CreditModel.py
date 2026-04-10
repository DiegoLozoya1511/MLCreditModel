import numpy as np
import pandas as pd


class CreditPortfolio:
    """
    Computes client-level lending decisions, break-even thresholds, and
    portfolio PnL for a given dataset split.

    Encapsulates the full decision pipeline: predicted probabilities →
    break-even thresholds → binary lending decisions → realized PnL vs
    benchmark PnL.

    Attributes
    ----------
    probabilities  : np.ndarray, predicted default probabilities (n_clients,)
    thresholds     : np.ndarray, break-even PD threshold per client (n_clients,)
    predictions    : np.ndarray, binary lending decisions, 1=lend (n_clients,)
    realized_pnl   : np.ndarray, PnL under model strategy (n_clients,)
    benchmark_pnl  : np.ndarray, PnL under lend-all benchmark (n_clients,)
    """

    def __init__(self, LGD: float = 0.75):
        """
        Parameters
        ----------
        LGD : Loss Given Default. Used in threshold computation and PnL
              calculation. Must be consistent with the value used in
              RatePricer to ensure coherent break-even derivation.
        """
        self.LGD = LGD

        self.probabilities = None
        self.thresholds = None
        self.predictions = None
        self.realized_pnl = None
        self.benchmark_pnl = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        model,
        X: pd.DataFrame,
        r: pd.Series,
        RB: np.ndarray,
        EAD: np.ndarray,
    ) -> "CreditPortfolio":
        """
        Compute predicted probabilities, break-even thresholds, and
        binary lending decisions.

        A client is approved (prediction=1) if and only if their predicted
        PD is below their individual break-even threshold, i.e. the expected
        PnL of lending is positive.

        Parameters
        ----------
        model : Fitted model with a predict_proba method. Compatible with
                both the raw sklearn interface and ModelFitter.predict_proba.
        X     : Feature matrix for the split being evaluated.
        r     : Client-level bucket interest rates, shape (n_clients,).
        RB    : Revolving balance (BILL_AMT1 - PAY_AMT1), shape (n_clients,).
        EAD   : Exposure at default (BILL_AMT1), shape (n_clients,).

        Returns
        -------
        self : Enables method chaining.
        """
        from fitter import ModelFitter

        if isinstance(model, ModelFitter):
            self.probabilities = model.predict_proba(X)
        else:
            self.probabilities = model.predict_proba(X)[:, 1]

        self.thresholds  = self._compute_thresholds(r, RB, EAD)
        self.predictions = (self.probabilities <= self.thresholds).astype(int)
        return self

    def compute_pnl(
        self,
        r: pd.Series,
        RB: np.ndarray,
        EAD: np.ndarray,
        y: np.ndarray,
    ) -> "CreditPortfolio":
        """
        Compute realized PnL per client under the model's lending strategy.

        PnL is defined as:
            0          if the client was rejected (prediction = 0)
            +RB * r    if the client was approved and did not default
            -EAD * LGD if the client was approved and defaulted

        Parameters
        ----------
        r    : Client-level bucket interest rates, shape (n_clients,).
        RB   : Revolving balance, shape (n_clients,).
        EAD  : Exposure at default, shape (n_clients,).
        y    : Actual default outcomes (1=default, 0=paid), shape (n_clients,).

        Returns
        -------
        self : Enables method chaining.
        """
        self._require_predictions()
        self.realized_pnl = self._pnl(r, RB, EAD, self.predictions, y)
        return self

    def compute_benchmark(
        self,
        r: pd.Series,
        RB: np.ndarray,
        EAD: np.ndarray,
        y: np.ndarray,
    ) -> "CreditPortfolio":
        """
        Compute realized PnL per client under a lend-to-all benchmark strategy.

        Parameters
        ----------
        r   : Client-level bucket interest rates, shape (n_clients,).
        RB  : Revolving balance, shape (n_clients,).
        EAD : Exposure at default, shape (n_clients,).
        y   : Actual default outcomes, shape (n_clients,).

        Returns
        -------
        self : Enables method chaining.
        """
        lend_all = np.ones(len(y), dtype=int)
        self.benchmark_pnl = self._pnl(r, RB, EAD, lend_all, y)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_thresholds(
        self,
        r: pd.Series,
        RB: np.ndarray,
        EAD: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the break-even PD threshold for each client.

        Derived from setting expected PnL = 0:
            PD* = (RB * r) / (RB * r + EAD * LGD)

        Clients with zero denominator receive a threshold of 0 (never lend).
        Result is clipped to [0, 1].
        """
        r = np.asarray(r,   dtype=float)
        RB = np.asarray(RB,  dtype=float)
        EAD = np.asarray(EAD, dtype=float)

        numerator = RB * r
        denominator = RB * r + EAD * self.LGD

        thresholds = np.divide(
            numerator, denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )
        return np.clip(thresholds, 0, 1)

    def _pnl(
        self,
        r: pd.Series,
        RB: np.ndarray,
        EAD: np.ndarray,
        pred: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Core PnL computation shared by compute_pnl and compute_benchmark.

        Parameters
        ----------
        pred : Lending decisions (1=lend, 0=reject).
        y    : Actual default outcomes (1=default, 0=paid).
        """
        r = np.asarray(r,   dtype=float)
        RB = np.asarray(RB,  dtype=float)
        EAD = np.asarray(EAD, dtype=float)

        return np.where(
            pred == 0,  0,
            np.where(
                y == 0, RB * r,
                -EAD * self.LGD
            )
        )

    def _require_predictions(self) -> None:
        if self.predictions is None:
            raise RuntimeError(
                "No predictions available. Call predict() before compute_pnl()."
            )
