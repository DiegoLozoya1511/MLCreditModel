import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV


class ModelFitter:
    """
    Handles XGBoost model training and probability calibration for the
    credit scoring pipeline.

    Training and calibration are intentionally separated into distinct
    methods to enforce the correct data discipline: the base model is
    fit on training data only, and calibration is applied on a held-out
    calibration subset to avoid data leakage.

    Attributes
    ----------
    model : fitted and calibrated CalibratedClassifierCV, or None if not
            yet fit and calibrated.
    """

    _BASE_MODEL_PATH = "Model/xgb_model.pkl"
    _CALIBRATED_MODEL_PATH = "Model/xgb_model_calibrated.pkl"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 3.5,
        min_child_weight: int = 5,
        reg_lambda: float = 2.0,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        All parameters map directly to XGBClassifier hyperparameters.
        Defaults replicate the configuration used in the original pipeline.
        """
        self._xgb_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            random_state=random_state,
            eval_metric="auc",
        )
        self.model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ModelFitter":
        """
        Fit the base XGBoost model on training data and persist it to disk.

        Parameters
        ----------
        X_train : Training feature matrix.
        y_train : Binary target vector (1 = default).

        Returns
        -------
        self : Enables method chaining.
        """
        base_model = XGBClassifier(**self._xgb_params)
        base_model.fit(X_train, y_train)
        joblib.dump(base_model, self._BASE_MODEL_PATH)
        print(f"Base model saved to {self._BASE_MODEL_PATH}")
        return self

    def calibrate(self, X_val_cal: pd.DataFrame, y_val_cal: pd.Series) -> "ModelFitter":
        """
        Load the base model, calibrate it on the calibration subset, and
        persist the calibrated model to disk.

        Isotonic calibration corrects the systematic overestimation of PD
        caused by scale_pos_weight without degrading ranking ability (AUC).

        Parameters
        ----------
        X_val_cal : Calibration feature matrix (held-out from validation set).
        y_val_cal : Binary target vector for the calibration subset.

        Returns
        -------
        self : Enables method chaining.
        """
        base_model = joblib.load(self._BASE_MODEL_PATH)
        calibrated = CalibratedClassifierCV(
            base_model, method="isotonic", cv=5)
        calibrated.fit(X_val_cal, y_val_cal)
        joblib.dump(calibrated, self._CALIBRATED_MODEL_PATH)
        self.model = calibrated
        print(f"Calibrated model saved to {self._CALIBRATED_MODEL_PATH}")
        return self

    def load(self) -> "ModelFitter":
        """
        Load the calibrated model from disk into self.model.

        Returns
        -------
        self : Enables method chaining.
        """
        self.model = joblib.load(self._CALIBRATED_MODEL_PATH)
        return self

    def predict_proba(self, X: pd.DataFrame):
        """
        Return predicted default probabilities for the positive class.

        Parameters
        ----------
        X : Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._require_model()
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call fit().calibrate() to train, "
                "or load() to restore a saved model."
            )
