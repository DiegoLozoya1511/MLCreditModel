import pandas as pd

from fitter import model_fitter
from utils import split_scale_pca
from CreditModel import get_predictions, compute_expected_pnl, realized_pnl
from visualization import plot_interest_rate_distribution, plot_thresholds_distribution, plot_roc_curve_train_val, plot_roc_curve_test, plot_pnl_distribution_comparisson, plot_confusion_matrix


def main():
    # === Load datasets ===
    data = pd.read_csv('Data/UCI_Credit_Card.csv')
    interest_rate_df = pd.read_csv('Data/InterestRate.csv')
    
    # === Define target and features ===
    target = 'default.payment.next.month'
    y = data[target]
    X = data.copy().drop(columns=['ID', target])
    
    # === Data preprocessing ===
    results = split_scale_pca(X, y, interest_rate_df['TotalInterestRate'])
    
    X_train, X_val, X_test = results["X_train"], results["X_val"], results["X_test"]
    y_train, y_val, y_test = results["y_train"], results["y_val"], results["y_test"]
    r_train, r_val, r_test = results["r_train"], results["r_val"], results["r_test"]
    EAD_train, EAD_val, EAD_test = results["EAD_train"], results["EAD_val"], results["EAD_test"]
    
    # === Model fitting ===
    # De-comment the line below to fit the model and save it to disk.
    #model_fitter(X_train, y_train)
    
    # === Credit Model ===
    # Load fitted model
    model = pd.read_pickle('Model/xgb_model.pkl')
    
    # --- Predictions ---
    train_probabilities, train_predictions, train_thresholds = get_predictions(model, X_train, r_train)
    val_probabilities, val_predictions, val_thresholds = get_predictions(model, X_val, r_val)
    #test_probabilities, test_predictions, test_thresholds = get_predictions(model, X_test, r_test)
    
    # --- Expected PnL Calculation ---
    train_expected_pnl = compute_expected_pnl(train_probabilities, r_train, EAD_train)
    val_expected_pnl = compute_expected_pnl(val_probabilities, r_val, EAD_val)
    #test_expected_pnl = compute_expected_pnl(test_probabilities, r_test, EAD_test)
    
    # --- Realized PnL Calculation ---
    train_realized_pnl = realized_pnl(r_train, EAD_train, train_predictions, y_train)
    val_realized_pnl = realized_pnl(r_val, EAD_val, val_predictions, y_val)
    #test_realized_pnl = realized_pnl(r_test, EAD_test, test_predictions, y_test)
    
    # NOT FINAL RESULTS - JUST QUICK CHECKS
    print(f'Total Expected PnL (Train): {train_expected_pnl.sum():,.2f}')
    print(f'Total Realized PnL (Train): {train_realized_pnl.sum():,.2f}')
    
    print(f'Total Expected PnL (Validation): {val_expected_pnl.sum():,.2f}')
    print(f'Total Realized PnL (Validation): {val_realized_pnl.sum():,.2f}')
    
    print('train:')
    approved = (train_predictions == 1)
    print(f"Default rate among approved: {y_train[approved].mean():.1%}")
    print(f"Overall default rate: {y_train.mean():.1%}")
    print(f"Approval rate: {train_predictions.mean():.1%}")
    print(f"Mean threshold: {train_thresholds.mean():.3f}")
    print(f"Clients above threshold: {(y_train > train_thresholds).mean():.1%}")

    print('validation:')
    approved = (val_predictions == 1)
    print(f"Default rate among approved: {y_val[approved].mean():.1%}")
    print(f"Overall default rate: {y_val.mean():.1%}")
    print(f"Approval rate: {val_predictions.mean():.1%})")
    print(f"Mean threshold: {val_thresholds.mean():.3f}")
    print(f"Clients above threshold: {(y_val > val_thresholds).mean():.1%}")
    
    # === Plots ===
    plot_interest_rate_distribution(interest_rate_df)
    
    plot_thresholds_distribution(train_thresholds, train_probabilities, 'Train')
    plot_thresholds_distribution(val_thresholds, val_probabilities, 'Validation')
    #plot_thresholds_distribution(test_thresholds, test_probabilities, 'Test')
    
    plot_roc_curve_train_val(y_train, train_probabilities, y_val, val_probabilities)
    #plot_roc_curve_test(y_test, test_probabilities)
    
    plot_pnl_distribution_comparisson(train_expected_pnl, train_realized_pnl, 'Train')    
    plot_pnl_distribution_comparisson(val_expected_pnl, val_realized_pnl, 'Validation')
    #plot_pnl_distribution_comparisson(test_expected_pnl, test_realized_pnl, 'Test')
    
    train_class_report = plot_confusion_matrix(train_predictions, y_train, 'Train')
    val_class_report = plot_confusion_matrix(val_predictions, y_val, 'Validation')
    #test_class_report = plot_confusion_matrix(test_predictions, y_test, 'Test')
    
    print(f"\nTrain Classification Report:\n{train_class_report}")
    print(f"\nValidation Classification Report:\n{val_class_report}")
    #print(f"\nTest Classification Report:\n{test_class_report}")

if __name__ == "__main__":
    main()
