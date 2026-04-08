import joblib
import pandas as pd

from utils import split_scale_pca, val_eval_cal_split
from fitter import model_fitter, model_calibrator
from CreditModel import get_predictions, expected_pnl, realized_pnl
from visualization import plot_interest_rate_distribution, plot_thresholds_distribution, plot_roc_curve_train_val, plot_roc_curve_test, plot_confusion_matrix, portfolio_metrics


def main():
    # === Load datasets ===
    data = pd.read_csv('Data/UCI_Credit_Card.csv')
    interest_rate_df = pd.read_csv('Data/InterestRate.csv')
    
    # === Define target and features ===
    target = 'default.payment.next.month'
    y = data[target]
    X = data.copy().drop(columns=['ID', target])
    
    # === Data preprocessing ===
    # Split data, scale features, and apply PCA. 
    # r, EAD and RV are passed to guarantee alignment but are not scaled or PCA-transformed.
    results = split_scale_pca(X, y, interest_rate_df['TotalInterestRate'])
    
    X_train, X_val, X_test = results["X_train"], results["X_val"], results["X_test"]
    y_train, y_val, y_test = results["y_train"], results["y_val"], results["y_test"]
    r_train, r_val, r_test = results["r_train"], results["r_val"], results["r_test"]
    EAD_train, EAD_val, EAD_test = results["EAD_train"], results["EAD_val"], results["EAD_test"]
    RV_train, RV_val, RV_test = results["RV_train"], results["RV_val"], results["RV_test"]
    
    # Split validation set into evaluation and calibration subsets for proper model calibration and evaluation without data leakage.
    val_results = val_eval_cal_split(X_val, y_val, r_val, EAD_val, RV_val)
    
    X_val_eval, X_val_cal = val_results["X_val_eval"], val_results["X_val_cal"]
    y_val_eval, y_val_cal = val_results["y_val_eval"], val_results["y_val_cal"]
    r_val_eval   = val_results["r_val_eval"]
    EAD_val_eval = val_results["EAD_val_eval"]
    RV_val_eval  = val_results["RV_val_eval"]
    
    # === Model fitting ===
    # De-comment the lines below to fit and calibrate the model and save it to disk.
    #model_fitter(X_train, y_train)
    #model_calibrator(X_val_cal, y_val_cal)
    
    # === Credit Model ===
    # Load fitted and calibrated model
    model = joblib.load('Model/xgb_model_calibrated.pkl')
    
    # --- Predictions ---
    train_probabilities, train_predictions, train_thresholds = get_predictions(model, X_train, r_train, RV_train, EAD_train)
    val_probabilities, val_predictions, val_thresholds = get_predictions(model, X_val_eval, r_val_eval, RV_val_eval, EAD_val_eval)
    #test_probabilities, test_predictions, test_thresholds = get_predictions(model, X_test, r_test, RV_test, EAD_test)
    
    # --- Expected PnL Calculation ---
    train_expected_pnl = expected_pnl(train_probabilities, r_train, RV_train, EAD_train)
    val_expected_pnl = expected_pnl(val_probabilities, r_val_eval, RV_val_eval, EAD_val_eval)
    #test_expected_pnl = expected_pnl(test_probabilities, r_test, RV_test, EAD_test)
    
    # --- Realized PnL Calculation ---
    train_realized_pnl = realized_pnl(r_train, RV_train, EAD_train, train_predictions, y_train)
    val_realized_pnl = realized_pnl(r_val_eval, RV_val_eval, EAD_val_eval, val_predictions, y_val_eval)
    #test_realized_pnl = realized_pnl(r_test, RV_test, EAD_test, test_predictions, y_test)
    
    # === Results ===
    # --- Plots ---
    plot_interest_rate_distribution(interest_rate_df)
    
    plot_thresholds_distribution(train_thresholds, train_probabilities, 'Train')
    plot_thresholds_distribution(val_thresholds, val_probabilities, 'Validation')
    #plot_thresholds_distribution(test_thresholds, test_probabilities, 'Test')
    
    plot_roc_curve_train_val(y_train, train_probabilities, y_val_eval, val_probabilities)
    #plot_roc_curve_test(y_test, test_probabilities)
    
    train_class_report = plot_confusion_matrix(train_predictions, y_train, 'Train')
    val_class_report = plot_confusion_matrix(val_predictions, y_val_eval, 'Validation')
    #test_class_report = plot_confusion_matrix(test_predictions, y_test, 'Test')
    
    # --- Prints ---
    # Classification reports
    print(f"\nTrain Classification Report:\n{'─' * 60}\n{train_class_report}")
    print(f"\nValidation Classification Report:\n{'─' * 60}\n{val_class_report}")
    #print(f"\nTest Classification Report:\n{test_class_report}")
    
    # Portfolio-level metrics
    portfolio_metrics(EAD_train, train_expected_pnl, train_realized_pnl, train_predictions, y_train, 'Train')
    portfolio_metrics(EAD_val_eval, val_expected_pnl, val_realized_pnl, val_predictions, y_val_eval, 'Validation')
    #portfolio_metrics(EAD_test, test_expected_pnl, test_realized_pnl, test_predictions, y_test, 'Test')

if __name__ == "__main__":
    main()
