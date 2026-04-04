import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


def calculate_interest_rate():
    
    # Read the dataset
    data = pd.read_csv('Data/UCI_Credit_Card.csv')
    
    # Split the dataset into features and target variable
    target = 'default.payment.next.month'
    y = data[target]
    
    X = data.copy().drop(columns=['ID', target])
        
    # Pipeline
    pipeline = Pipeline([
        # Scaling
        ('scaler', StandardScaler()),
        # Logistic Regression
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    
    # Calculate PD using Out-of-Fold predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    PD_oof = np.zeros(len(X))
    
    for train_idx, val_idx in skf.split(X, y):
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        PD_oof[val_idx] = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        
    X['PD_i'] = PD_oof
    
    # Calculate risk premium using the formula: Risk Premium = (PD_i * LGD) / (1 - PD_i)
    LGD = 0.75 # Industry asumption

    risk_premium = (X['PD_i'] * LGD) / (1 - X['PD_i'].clip(upper=0.9999))
    
    # Create interest rate DataFrame
    BASE_RATE = 0.0675
    INFLATION_RATE = 0.0450
    LIQUIDITY_PREMIUM = 0.02
    ADMIN_COSTS = 0.07
    OPERATING_COSTS = 0.05
    
    REAL_RATE = BASE_RATE - INFLATION_RATE
    
    InterestRate = pd.DataFrame({
        'Real Rate': REAL_RATE,
        'Inflation Rate': INFLATION_RATE,
        'Risk Premium': risk_premium,
        'Liquidity Premium': LIQUIDITY_PREMIUM,
        'Admin Costs': ADMIN_COSTS,
        'Operating Costs': OPERATING_COSTS,
    })
    
    # CAT ceiling based on Banxico official data (Ualá, cards > $15,000 limit)
    CAT_CAP = 1.39  # 139%

    # Implied annual rate ceiling (CAT - fixed costs approximation)
    # CAT ≈ r_i + fixed costs overhead, so r_i cap ≈ CAT_CAP - fixed costs
    FIXED_COSTS = INFLATION_RATE + LIQUIDITY_PREMIUM + ADMIN_COSTS + OPERATING_COSTS
    RATE_CAP = CAT_CAP - FIXED_COSTS 
    
    InterestRate['TotalInterestRate'] = InterestRate.sum(axis=1).clip(upper=RATE_CAP)
    
    # Write results to CSV
    InterestRate.to_csv('Data/InterestRate.csv', index=False)
        
    print("Out-of-Fold Predicted Probabilities (PD):")
    print(X.head())
    print("\nAverage PD:", np.mean(X['PD_i'].mean()))
    print("\nAverage Risk Premium:", np.mean(risk_premium))
    print("\nInterest Rate DataFrame:")
    print(InterestRate.head())
    print("\nMax Total Interest Rate:", InterestRate['TotalInterestRate'].max())
    print("\nMin Total Interest Rate:", InterestRate['TotalInterestRate'].min())
    print("\nAverage Total Interest Rate:", InterestRate['TotalInterestRate'].mean())
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(InterestRate['TotalInterestRate'], fill=True, color='navy')
    plt.axvline(InterestRate['TotalInterestRate'].mean(), color='indianred', 
                linestyle='--', label=f'Mean Total Interest Rate: {InterestRate['TotalInterestRate'].mean():.2%}')
    plt.title('Distribution of Total Interest Rates')
    plt.xlabel('Total Interest Rate')
    plt.ylabel('Density')
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    calculate_interest_rate()