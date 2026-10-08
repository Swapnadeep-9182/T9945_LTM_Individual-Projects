import pandas as pd
import numpy as np
import os

def explore_and_clean_data():
    print("="*50)
    print("📊 PHASE 1: DATA EXPLORATION & CLEANING")
    print("="*50)

    # 1. Download or Load the Dataset
    file_path = "data/Telco-Customer-Churn.csv"
    data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    if not os.path.exists(file_path):
        print("[!] Dataset not found locally. Downloading from IBM repository...")
        df = pd.read_csv(data_url)
        df.to_csv(file_path, index=False)
        print(f"[✓] Dataset saved to {file_path}")
    else:
        df = pd.read_csv(file_path)
        print(f"[✓] Dataset loaded from {file_path}")

    # 2. Basic Information
    print("\n📝 Dataset Shape:", df.shape)
    print("Columns:", list(df.columns))

    # 3. Fix the 'TotalCharges' Bug
    # TotalCharges is formatted as an 'object' (string) because of hidden blank spaces. 
    # We must force it to numerical and handle the blanks.
    print("\n🛠️ Cleaning 'TotalCharges' column...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Check how many missing values were created by the coercion
    missing_charges = df['TotalCharges'].isnull().sum()
    print(f"   Found {missing_charges} missing values after conversion. Dropping them.")
    
    # Drop those 11 rows (out of 7043, it's safe to drop)
    df.dropna(subset=['TotalCharges'], inplace=True)

    # 4. Target Variable Distribution
    print("\n🎯 Target Variable ('Churn') Distribution:")
    churn_counts = df['Churn'].value_counts()
    churn_percentages = df['Churn'].value_counts(normalize=True) * 100
    
    for label, count in churn_counts.items():
        percent = churn_percentages[label]
        print(f"   - {label}: {count} customers ({percent:.2f}%)")

    # Save the cleaned dataset for the next step
    cleaned_path = "data/Telco-Customer-Churn-Cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"\n[✓] Cleaned dataset saved to {cleaned_path}")
    print("="*50)

if __name__ == "__main__":
    explore_and_clean_data()