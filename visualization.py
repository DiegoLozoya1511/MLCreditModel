import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.loc'] = 'best'

blue_colors = [
    "#D9EDFF",
    "#C1E2FF",
    "#A8D5FF",
    "#8AC6FF",
    "#6BB6FF",
    "#4A9EE5",
    "#2E86DE",
    "#0066C0",
    "#0055A5",
    "#003D82"
]

blue_scale = mcolors.LinearSegmentedColormap.from_list(
    "blue_scale",
    [blue_colors[0], blue_colors[-1]]
)

complementary_colors = [
    "#FFEBD9",
    "#FFDEC1",
    "#FFD2A8",
    "#FFC38A",
    "#FFB56B",
    "#E5904A",
    "#DE862E",
    "#C05A00",
    "#A55000",
    "#824500"
]  


def plot_interest_rate_distribution(interest_rate_df: pd.DataFrame) -> None:
    """
    Plot the distribution of continuous interest rates colored by bucket.

    Parameters:
        interest_rate_df: pandas DataFrame containing 'TotalInterestRate', 'BucketRate' and 'Bucket' columns.
    """
    buckets = sorted(interest_rate_df['Bucket'].unique())
    colors  = [blue_colors[int(i * (len(blue_colors) - 1) / (len(buckets) - 1))] for i in range(len(buckets))]
    colors = colors[::-1]

    plt.figure()
    for bucket, color in zip(buckets, colors):
        mask        = interest_rate_df['Bucket'] == bucket
        bucket_data = interest_rate_df.loc[mask, 'TotalInterestRate']  # continuous rate
        bucket_rate = interest_rate_df.loc[mask, 'BucketRate'].iloc[0] # centroid for label

        sns.kdeplot(bucket_data, fill=True, color=color, alpha=0.5,
                    label=f'Bucket {bucket}: {bucket_rate:.2%}')

    plt.axvline(interest_rate_df['BucketRate'].mean(), color='dimgray',
                linestyle='--', linewidth=0.8,
                label=f'Mean Rate: {interest_rate_df["BucketRate"].mean():.2%}')
    plt.title('Distribution of Total Interest Rates by Bucket')
    plt.xlabel('Total Interest Rate')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def plot_thresholds_distribution(thresholds: pd.Series, y_proba: np.ndarray, set_name: str) -> None:
    """
    Plot the distribution of break-even thresholds and model predicted probabilities for a given dataset (train/validation/test).

    Parameters:
        thresholds: pandas Series of break-even PD thresholds for each client.
        y_proba: numpy array of predicted probabilities of default for each client.
        set_name: string indicating the dataset name (e.g., 'Train', 'Validation', 'Test') for labeling the plot.
    """
    y_proba = pd.Series(y_proba)

    plt.figure()
    sns.kdeplot(y_proba, fill=True, alpha=0.5,
                color=blue_colors[3],  label=f'Model PD')
    plt.axvline(y_proba.mean(),    color=complementary_colors[3],  linestyle='--',
                linewidth=0.9, label=f'Mean PD: {y_proba.mean():.2%}')
    sns.kdeplot(thresholds, fill=True, alpha=0.5,
                color=blue_colors[-1], label=f'Break-even Threshold')
    plt.axvline(thresholds.mean(), color=complementary_colors[-1], linestyle='--',
                linewidth=0.9, label=f'Mean Threshold: {thresholds.mean():.2%}')
    plt.title(f'{set_name} Distribution of PD vs Break-even Threshold')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def plot_roc_curve_train_val(y_true_train: np.ndarray, y_proba_train: np.ndarray, y_true_val: np.ndarray, y_proba_val: np.ndarray) -> None:
    """
    Plot ROC curves for both train and validation sets on the same plot for comparison.

    Parameters:
        y_true_train: numpy array of true binary labels for the training set.
        y_proba_train: numpy array of predicted probabilities for the positive class for the training set.
        y_true_val: numpy array of true binary labels for the validation set.
        y_proba_val: numpy array of predicted probabilities for the positive class for the validation set.
    """
    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_proba_train)
    fpr_val,   tpr_val,   _ = roc_curve(y_true_val,   y_proba_val)

    auc_train = roc_auc_score(y_true_train, y_proba_train)
    auc_val = roc_auc_score(y_true_val,   y_proba_val)

    plt.figure()
    plt.plot(fpr_train, tpr_train,
             color=blue_colors[-1], label=f'Train: AUC = {auc_train:.4f}')
    plt.plot(fpr_val,   tpr_val,
             color=blue_colors[2],  label=f'Validation: AUC = {auc_val:.4f}')
    plt.fill_between(fpr_train, tpr_train, alpha=0.25, color=blue_colors[-1])
    plt.fill_between(fpr_val, tpr_val, alpha=0.25, color=blue_colors[2])
    plt.plot([0, 1], [0, 1], color='dimgray', linestyle='--',
             linewidth=0.8, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Train and Validation ROC Curves')
    plt.legend()
    plt.show()


def plot_roc_curve_test(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """
    Plot ROC curve for the test set.

    Parameters:
        y_true: numpy array of true binary labels for the test set.
        y_proba: numpy array of predicted probabilities for the positive class for the test set.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, color=blue_colors[-1], label=f'Test AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], color='dimgray', linestyle='--',
             linewidth=0.8, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.15, color=blue_colors[-1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend()
    plt.show()


def plot_confusion_matrix(predicted: np.ndarray, real: pd.Series, set: str) -> str:
    """
    Plot confusion matrix and return classification report.

    Parameters:
        predicted: numpy array of predicted binary classifications (0 or 1).
        real: pandas Series of true binary labels (0 or 1).
        set: string indicating the dataset name (e.g., 'Train', 'Validation', 'Test') for labeling the plot.
    """
    real_lend = 1 - real  # Invert real labels to match "Lend" = 1, "No Lend" = 0 for better interpretability in the confusion matrix

    acc = accuracy_score(real_lend, predicted)
    class_report = classification_report(
        real_lend, predicted)

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay.from_predictions(
        real_lend,
        predicted,
        cmap=blue_scale,
        colorbar=True,
        values_format='d',
        ax=ax
    )

    ax.set_title(f'{set} Confusion Matrix - Accuracy: {acc:.4f}')
    ax.grid(False)

    plt.show()

    return class_report


def interest_rate_summary(r_summary: pd.DataFrame, total_interest_rates: pd.DataFrame) -> None:
    """
    Print a summary of interest rate buckets and their statistics.

    Parameters:
        r_summary: DataFrame containing summary statistics for each interest rate bucket (e.g., mean PD, mean risk premium, mean total interest rate).
        total_interest_rates: DataFrame containing the total interest rates and their components for all clients, used to calculate overall means.
    """
    print(f"\nInterest Rate Bucket Summary:\n{'─' * 105}")
    print(r_summary.to_string(index=False))
    print(f'\nMean PD: {total_interest_rates["PD_i"].mean():.2%}')
    print(
        f'Mean Risk Premium: {(total_interest_rates["Risk Premium"]).mean():.2%}')
    print(
        f'Mean Interest Rate: {total_interest_rates["BucketRate"].mean():.2%}')


def portfolio_metrics(EAD: pd.Series, benchmark_pnl: np.ndarray, realized_pnl: np.ndarray, pred: np.ndarray, y_true: np.ndarray, set_name: str) -> None:
    """
    Compute and print portfolio-level metrics for credit model evaluation.

    Parameters:
        EAD          : exposure at default (BILL_AMT1), shape (n_clients,)
        expected_pnl : expected profit and loss per client, shape (n_clients,)
        realized_pnl : realized profit and loss per client, shape (n_clients,)
        pred         : lending decision (1=lend, 0=reject), shape (n_clients,)
        y_true       : actual default outcome (1=default, 0=paid), shape (n_clients,)
        set_name     : label for the dataset (e.g., 'Train', 'Validation', 'Test')
    """
    EAD = np.asarray(EAD,    dtype=float)
    pred = np.asarray(pred,   dtype=int)
    y_true = np.asarray(y_true, dtype=int)

    total_clients = len(pred)
    clients_approved = (pred == 1).sum()
    approval_rate = clients_approved / total_clients

    defaulters = ((pred == 1) & (y_true == 1)).sum()
    default_rate = defaulters / clients_approved if clients_approved > 0 else 0.0

    total_benchmark_pnl = benchmark_pnl.sum()
    total_realized_pnl = realized_pnl.sum()
    pnl_uplift = (total_realized_pnl - total_benchmark_pnl) / \
        abs(total_benchmark_pnl) if total_benchmark_pnl != 0 else 0.0

    capital_total = EAD.sum()
    capital_deployed = EAD[pred == 1].sum()
    capital_rate = capital_deployed / capital_total

    benchmark_roc = total_benchmark_pnl / capital_total if capital_total > 0 else 0.0
    realized_roc = total_realized_pnl / \
        capital_deployed if capital_deployed > 0 else 0.0

    bench_str = f"${total_benchmark_pnl:,.2f}"
    realized_str = f"${total_realized_pnl:,.2f}"

    W = 13  # value column width

    print(f"\n{set_name} Portfolio Metrics")
    print(f"{'─' * 72}")
    print(f"{'Total clients:':<30} {total_clients:>{W},}")
    print(f"{'Clients approved:':<30} {clients_approved:>{W},}     ({approval_rate:.2%})")
    print(f"{'Defaulters:':<30} {defaulters:>{W},}     ({default_rate:.2%} of approved)")
    print(f"{'Benchmark PnL (lend all):':<30} {bench_str:>{W+3}}")
    print(f"{'Realized PnL (model):':<30} {realized_str:>{W+3}}  ({pnl_uplift:+.2%} vs benchmark)")
    print(f"{'Capital total:':<30} {capital_total:>{W},.0f}")
    print(f"{'Capital deployed:':<30} {capital_deployed:>{W},.0f}     ({capital_rate:.2%})")
    print(f"{'Benchmark Return on Capital:':<30} {benchmark_roc:>{W+4}.2%}")
    print(f"{'Realized Return on Capital:':<30} {realized_roc:>{W+4}.2%}")
