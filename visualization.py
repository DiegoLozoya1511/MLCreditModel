import seaborn as sns
import matplotlib.pyplot as plt


def plot_interest_rate_distribution(interest_rate_df):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(interest_rate_df['TotalInterestRate'], fill=True, color='navy')
    plt.axvline(interest_rate_df['TotalInterestRate'].mean(), color='indianred', 
                linestyle='--', label=f'Mean Total Interest Rate: {interest_rate_df['TotalInterestRate'].mean():.2%}')
    plt.title('Distribution of Total Interest Rates')
    plt.xlabel('Total Interest Rate')
    plt.ylabel('Density')
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()