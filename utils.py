import pandas as pd

from sklearn.model_selection import train_test_split


class DataPipeline:
    """
    Handles stratified train / validation / test splitting for the credit
    scoring pipeline.

    The validation set is further divided into an evaluation subset (used
    for model assessment) and a calibration subset (used for probability
    calibration), keeping both partitions leak-free.

    Attributes
    ----------
    X_train, X_val_eval, X_val_cal, X_test : pd.DataFrame
    y_train, y_val_eval, y_val_cal, y_test : np.ndarray
    EAD_train, EAD_val_eval, EAD_val_cal, EAD_test : np.ndarray
    RB_train,  RB_val_eval,  RB_val_cal,  RB_test  : np.ndarray
    """

    def __init__(
        self,
        test_size: float = 0.05,
        val_size: float = 0.25,
        eval_size: float = 0.30,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        test_size    : Proportion of the full dataset reserved for test.
        val_size     : Proportion of the full dataset reserved for validation.
        eval_size    : Proportion of the validation set reserved for evaluation.
                       The remainder becomes the calibration subset.
        random_state : Random seed for reproducibility.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.eval_size = eval_size
        self.random_state = random_state

        # Public attributes populated by fit_transform()
        self.X_train = self.X_val_eval = self.X_val_cal = self.X_test = None
        self.y_train = self.y_val_eval = self.y_val_cal = self.y_test = None
        self.EAD_train = self.EAD_val_eval = self.EAD_val_cal = self.EAD_test = None
        self.RB_train = self.RB_val_eval = self.RB_val_cal = self.RB_test = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> "DataPipeline":
        """
        Execute the full split pipeline and populate all attributes.

        Parameters
        ----------
        X : Feature matrix. Must contain 'BILL_AMT1' and 'PAY_AMT1'.
        y : Binary target vector (1 = default).

        Returns
        -------
        self : Enables method chaining.
        """
        EAD = X["BILL_AMT1"]
        RB = (X["BILL_AMT1"] - X["PAY_AMT1"]).clip(lower=0)

        X_train, X_val, X_test, y_train, y_val, y_test, EAD_train, EAD_val, EAD_test, RB_train, RB_val, RB_test = \
            self._primary_split(X, y, EAD, RB)

        X_val_eval, X_val_cal, y_val_eval, y_val_cal, EAD_val_eval, EAD_val_cal, RB_val_eval, RB_val_cal = \
            self._val_split(X_val, y_val, EAD_val, RB_val)

        self.X_train,   self.X_val_eval,   self.X_val_cal,   self.X_test = X_train,   X_val_eval,   X_val_cal,   X_test
        self.y_train,   self.y_val_eval,   self.y_val_cal,   self.y_test = y_train,   y_val_eval,   y_val_cal,   y_test
        self.EAD_train, self.EAD_val_eval, self.EAD_val_cal, self.EAD_test = EAD_train, EAD_val_eval, EAD_val_cal, EAD_test
        self.RB_train,  self.RB_val_eval,  self.RB_val_cal,  self.RB_test = RB_train,  RB_val_eval,  RB_val_cal,  RB_test

        self._print_summary()
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _primary_split(self, X, y, EAD, RB):
        """Train / val / test split with stratification."""
        val_ratio_adjusted = self.val_size / (1.0 - self.test_size)

        X_temp, X_test, y_temp, y_test, EAD_temp, EAD_test, RB_temp, RB_test = \
            train_test_split(
                X, y, EAD, RB,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state,
            )

        X_train, X_val, y_train, y_val, EAD_train, EAD_val, RB_train, RB_val = \
            train_test_split(
                X_temp, y_temp, EAD_temp, RB_temp,
                test_size=val_ratio_adjusted,
                stratify=y_temp,
                random_state=self.random_state,
            )

        return (
            X_train, X_val, X_test,
            y_train.values, y_val.values, y_test.values,
            EAD_train.values, EAD_val.values, EAD_test.values,
            RB_train.values, RB_val.values, RB_test.values,
        )

    def _val_split(self, X_val, y_val, EAD_val, RB_val):
        """Split the validation set into evaluation and calibration subsets."""
        X_val_eval, X_val_cal, y_val_eval, y_val_cal, EAD_val_eval, EAD_val_cal, RB_val_eval, RB_val_cal = \
            train_test_split(
                X_val, y_val, EAD_val, RB_val,
                test_size=self.eval_size,
                stratify=y_val,
                random_state=self.random_state,
            )

        return (
            X_val_eval, X_val_cal,
            y_val_eval, y_val_cal,
            EAD_val_eval, EAD_val_cal,
            RB_val_eval, RB_val_cal,
        )

    def _print_summary(self) -> None:
        n_train = len(self.y_train)
        n_eval = len(self.y_val_eval)
        n_cal = len(self.y_val_cal)
        n_test = len(self.y_test)
        print(
            f"\nSizes — train: {n_train}, "
            f"val_eval: {n_eval}, val_cal: {n_cal}, "
            f"test: {n_test}"
        )
