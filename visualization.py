import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import accuracy_score, classification_report
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
    sns.kdeplot(interest_rate_df['TotalInterestRate'], fill=True, color=blue_colors[-1])
    plt.axvline(interest_rate_df['TotalInterestRate'].mean(), color='dimgray', 
                linestyle='--', label=f'Mean Total Interest Rate: {interest_rate_df['TotalInterestRate'].mean():.2%}')
    plt.title('Distribution of Total Interest Rates')
    plt.xlabel('Total Interest Rate')
    plt.ylabel('Density')
    plt.grid()
    plt.legend()
    plt.show()