import pandas as pd
import joblib

from xgboost import XGBClassifier

def model_fitter(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Fit an XGBoost model to the training data and save it to disk.
    
    Parameters:
        X_train: pandas DataFrame of training features
        y_train: pandas Series of training target variable (binary)
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'Model/xgb_model.pkl')
