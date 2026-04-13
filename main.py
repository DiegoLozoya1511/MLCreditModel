import pandas as pd

from fitter import ModelFitter
from InterestRate import RatePricer
from DataSplitter import DataPipeline
from CreditModel import CreditPortfolio
from visualization import (
    plot_interest_rate_distribution,
    plot_thresholds_distribution,
    plot_roc_curve_train_val,
    plot_confusion_matrix,
    interest_rate_summary,
    plot_roc_curve_test,
    portfolio_metrics,
)


def main():

    # ==========================================================================
    # 1. DATA
    # ==========================================================================

    data = pd.read_csv("Data/UCI_Credit_Card.csv")
    target = "default.payment.next.month"

    y = data[target]
    X = data.drop(columns=["ID", target])

    # Split into train / val_eval / val_cal / test.
    # EAD and RB are derived internally from BILL_AMT1 and PAY_AMT1.
    pipeline = DataPipeline().fit_transform(X, y)

    # ==========================================================================
    # 2. MODEL
    # ==========================================================================

    # Set RETRAIN = True to fit and calibrate from scratch.
    # Set RETRAIN = False to load the persisted calibrated model from disk.
    RETRAIN = False

    fitter = ModelFitter()
    if RETRAIN:
        fitter.fit(pipeline.X_train, pipeline.y_train) \
              .calibrate(pipeline.X_val_cal, pipeline.y_val_cal)
    else:
        fitter.load()

    # ==========================================================================
    # 3. INTEREST RATES
    # ==========================================================================

    # Price each split independently to avoid information leakage across sets.
    pricer_train = RatePricer().price(pd.Series(fitter.predict_proba(pipeline.X_train)))
    pricer_val = RatePricer().price(pd.Series(fitter.predict_proba(pipeline.X_val_eval)))
    pricer_test = RatePricer().price(pd.Series(fitter.predict_proba(pipeline.X_test)))

    # Aggregate rates and summary across all splits for reporting.
    total_rates, r_summary = RatePricer.combined_summary(
        [pricer_train, pricer_val, pricer_test]
    )

    # ==========================================================================
    # 4. PORTFOLIO — TRAIN
    # ==========================================================================

    train_portfolio = (
        CreditPortfolio()
        .predict(fitter, pipeline.X_train, pricer_train.bucket_rates, pipeline.RB_train, pipeline.EAD_train)
        .compute_pnl(pricer_train.bucket_rates, pipeline.RB_train, pipeline.EAD_train, pipeline.y_train)
        .compute_benchmark(pricer_train.bucket_rates, pipeline.RB_train, pipeline.EAD_train, pipeline.y_train)
    )

    # ==========================================================================
    # 5. PORTFOLIO — VALIDATION
    # ==========================================================================

    val_portfolio = (
        CreditPortfolio()
        .predict(fitter, pipeline.X_val_eval, pricer_val.bucket_rates, pipeline.RB_val_eval, pipeline.EAD_val_eval)
        .compute_pnl(pricer_val.bucket_rates, pipeline.RB_val_eval, pipeline.EAD_val_eval, pipeline.y_val_eval)
        .compute_benchmark(pricer_val.bucket_rates, pipeline.RB_val_eval, pipeline.EAD_val_eval, pipeline.y_val_eval)
    )
    
    # ==========================================================================
    # 6. PORTFOLIO — TEST
    # ==========================================================================
    
    test_portfolio = (
        CreditPortfolio()
        .predict(fitter, pipeline.X_test, pricer_test.bucket_rates, pipeline.RB_test, pipeline.EAD_test)
        .compute_pnl(pricer_test.bucket_rates, pipeline.RB_test, pipeline.EAD_test, pipeline.y_test)
        .compute_benchmark(pricer_test.bucket_rates, pipeline.RB_test, pipeline.EAD_test, pipeline.y_test)
    )

    # ==========================================================================
    # 7. RESULTS
    # ==========================================================================

    # --- Interest rates ---
    plot_interest_rate_distribution(total_rates)
    interest_rate_summary(r_summary, total_rates)

    # --- Threshold distributions ---
    plot_thresholds_distribution(
        train_portfolio.thresholds, train_portfolio.probabilities, "Train")
    plot_thresholds_distribution(
        val_portfolio.thresholds,   val_portfolio.probabilities,   "Validation")
    plot_thresholds_distribution(
        test_portfolio.thresholds,  test_portfolio.probabilities,  "Test")

    # --- ROC curves ---
    plot_roc_curve_train_val(
        pipeline.y_train,    train_portfolio.probabilities,
        pipeline.y_val_eval, val_portfolio.probabilities,
    )
    plot_roc_curve_test(
        pipeline.y_test, test_portfolio.probabilities,
    )

    # --- Confusion matrices ---
    train_class_report = plot_confusion_matrix(
        train_portfolio.predictions, pipeline.y_train,    "Train")
    val_class_report = plot_confusion_matrix(
        val_portfolio.predictions,   pipeline.y_val_eval, "Validation")
    test_class_report = plot_confusion_matrix(
        test_portfolio.predictions,  pipeline.y_test,     "Test")

    # --- Classification reports ---
    print(f"\nTrain Classification Report:\n{'─' * 55}\n{train_class_report}")
    print(
        f"\nValidation Classification Report:\n{'─' * 55}\n{val_class_report}")
    print(f"\nTest Classification Report:\n{'─' * 55}\n{test_class_report}")

    # --- Portfolio metrics ---
    portfolio_metrics(
        pipeline.EAD_train, train_portfolio.benchmark_pnl,
        train_portfolio.realized_pnl, train_portfolio.predictions,
        pipeline.y_train, "Train",
    )
    portfolio_metrics(
        pipeline.EAD_val_eval, val_portfolio.benchmark_pnl,
        val_portfolio.realized_pnl, val_portfolio.predictions,
        pipeline.y_val_eval, "Validation",
    )
    portfolio_metrics(
        pipeline.EAD_test, test_portfolio.benchmark_pnl,
        test_portfolio.realized_pnl, test_portfolio.predictions,
        pipeline.y_test, "Test",
    )

    print()


if __name__ == "__main__":
    main()
