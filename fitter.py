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
