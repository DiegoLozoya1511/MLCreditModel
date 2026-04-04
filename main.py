import pandas as pd

from visualization import plot_interest_rate_distribution


def main():
    # Load datasets
    data = pd.read_csv('Data/UCI_Credit_Card.csv')
    interest_rate_df = pd.read_csv('Data/InterestRate.csv')
    
    # Plots
    plot_interest_rate_distribution(interest_rate_df)
    

if __name__ == "__main__":
    main()