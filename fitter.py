import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier


def model_fitter(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Fit an XGBoost model to the training data and save it to disk.

    Parameters:
        X_train: pandas DataFrame of training features
        y_train: pandas Series of training target variable (binary)
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3.5,
        min_child_weight=5,
        reg_lambda=2.0,
        random_state=42,
        eval_metric='auc',
    )
    model.fit(X_train, y_train)

    joblib.dump(model, 'Model/xgb_model.pkl')


def model_calibrator(X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """
    Load a fitted XGBoost model, calibrate it on the validation set, and save it to disk.
    Calibration corrects systematic overestimation of PD caused by scale_pos_weight
    without affecting the model's ranking ability (AUC).

    Parameters:
        X_val: pandas DataFrame of validation features
        y_val: pandas Series of validation target variable (binary)
    """
    model = joblib.load('Model/xgb_model.pkl')

    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated_model.fit(X_val, y_val)

    joblib.dump(calibrated_model, 'Model/xgb_model_calibrated.pkl')
