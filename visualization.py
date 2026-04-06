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


def plot_interest_rate_distribution(interest_rate_df):
    plt.figure()
    sns.kdeplot(interest_rate_df['TotalInterestRate'], fill=True, color=blue_colors[-1], label='Total Interest Rate')
    plt.axvline(interest_rate_df['TotalInterestRate'].mean(), color='dimgray', 
                linestyle='--', label=f'Mean Total Interest Rate: {interest_rate_df['TotalInterestRate'].mean():.2%}')
    plt.title('Distribution of Total Interest Rates')
    plt.xlabel('Total Interest Rate')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    
def plot_thresholds_distribution(thresholds: pd.Series, y_proba: np.ndarray, set_name: str):
    y_proba = pd.Series(y_proba)

    plt.figure()
    sns.kdeplot(thresholds, fill=True, color=blue_colors[-1],  label=f'Break-even Threshold (mean={thresholds.mean():.2%})')
    sns.kdeplot(y_proba,    fill=True, color=blue_colors[2],   label=f'Model PD (mean={y_proba.mean():.2%})')
    plt.axvline(thresholds.mean(), color='dimgray',   linestyle='--', linewidth=0.8, label=f'Mean Threshold: {thresholds.mean():.2%}')
    plt.axvline(y_proba.mean(),    color='black',     linestyle='--', linewidth=0.8, label=f'Mean PD: {y_proba.mean():.2%}')
    plt.title(f'{set_name} Distribution of PD vs Break-even Threshold')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    

def plot_roc_curve_train_val(y_true_train: np.ndarray, y_proba_train: np.ndarray ,y_true_val: np.ndarray, y_proba_val: np.ndarray):

    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_proba_train)
    fpr_val,   tpr_val,   _ = roc_curve(y_true_val,   y_proba_val)

    auc_train = roc_auc_score(y_true_train, y_proba_train)
    auc_val   = roc_auc_score(y_true_val,   y_proba_val)

    plt.figure()
    plt.plot(fpr_train, tpr_train, color=blue_colors[-1], label=f'Train: AUC = {auc_train:.4f}')
    plt.plot(fpr_val,   tpr_val,   color=blue_colors[2],  label=f'Validation: AUC = {auc_val:.4f}')
    plt.fill_between(fpr_train, tpr_train, alpha=0.15, color=blue_colors[-1])
    plt.fill_between(fpr_val, tpr_val, alpha=0.3, color=blue_colors[2])
    plt.plot([0, 1], [0, 1], color='dimgray', linestyle='--', linewidth=0.8, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Train and Validation ROC Curves')
    plt.legend()
    plt.show()
    
    
def plot_roc_curve_test(y_true: np.ndarray, y_proba: np.ndarray):

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, color=blue_colors[-1], label=f'Test AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], color='dimgray', linestyle='--', linewidth=0.8, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.15, color=blue_colors[-1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend()
    plt.show()
    

def plot_pnl_distribution_comparisson(expected_pnl: pd.Series, realized_pnl: pd.Series, set_name: str, clip_percentile: float = 0.99):
    # Clip to percentile range to remove extreme tails
    upper = np.percentile(
        np.concatenate([expected_pnl, realized_pnl]),
        clip_percentile * 100
    )
    lower = np.percentile(
        np.concatenate([expected_pnl, realized_pnl]),
        (1 - clip_percentile) * 100
    )

    expected_clipped = expected_pnl.clip(lower, upper)
    realized_clipped = realized_pnl.clip(lower, upper)
    
    plt.figure()
    sns.kdeplot(expected_clipped, fill=True, color=blue_colors[2],  label=f'Total expected PnL: ${expected_clipped.sum():,.2f}')
    sns.kdeplot(realized_clipped, fill=True, color=blue_colors[-1], label=f'Total realized PnL: ${realized_clipped.sum():,.2f}')
    plt.axvline(0, color='dimgray', linestyle='--', linewidth=0.8, label=f'Zero')
    plt.title(f'{set_name} Expected vs Realized Distribution of PnL')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def plot_confusion_matrix(predicted: np.ndarray, real: pd.Series, set: str) -> str:
    real_lend = 1 - real # Invert real labels to match "Lend" = 1, "No Lend" = 0 for better interpretability in the confusion matrix
    
    acc = accuracy_score(real_lend, predicted)
    class_report = classification_report(
        real_lend, predicted)#, target_names=['No Lend', 'Lend'])

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
