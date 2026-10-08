import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import os

def train_baseline_model():
    print("="*50)
    print("🚀 PHASE 1: MODEL TRAINING & PIPELINE EXPORT")
    print("="*50)

    # 1. Load Cleaned Data
    df = pd.read_csv("data/Telco-Customer-Churn-Cleaned.csv")
    
    # Drop customerID as it is not a predictive feature
    df.drop(columns=['customerID'], inplace=True)

    # Convert Target 'Churn' to binary (Yes=1, No=0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Define Features (X) and Target (y)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # 2. Train/Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Define Preprocessing Steps
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # This handles scaling numbers and one-hot encoding text automatically
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # 4. Define XGBoost Model
    # scale_pos_weight forces the model to care equally about the minority class (Churn=Yes)
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    model = xgb.XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    # 5. Create Full Pipeline
    # We bundle the preprocessing and the model together so we only have to export one file!
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    # 6. Train the Model
    print("[⏳] Training XGBoost model with preprocessing pipeline...")
    pipeline.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n📊 Model Evaluation:")
    print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)']))
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"🌟 ROC-AUC Score: {auc:.4f}")

    # 8. Export Pipeline for Real-Time Streaming
    joblib.dump(pipeline, "models/xgboost_pipeline.pkl")
    print("\n[✓] Full pipeline (preprocessing + model) saved to models/xgboost_pipeline.pkl")
    print("="*50)

if __name__ == "__main__":
    train_baseline_model()