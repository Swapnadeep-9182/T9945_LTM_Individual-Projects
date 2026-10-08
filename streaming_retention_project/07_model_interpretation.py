import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("🔍 LAUNCHING EXPLAINABLE AI (XAI) MODULE")
print("="*50)

# ১. ডেটা এবং প্রি-ট্রেইনড মডেল লোড করা
print("[⏳] Loading saved pipeline and cleaned data...")
df = pd.read_csv("data/Telco-Customer-Churn-Cleaned.csv")
pipeline = joblib.load("models/xgboost_pipeline.pkl")

# ডেটা প্রিপারেশন
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)
X = df.drop(columns=['Churn'])

# Convert integer columns to float for PDP compatibility
int_cols = X.select_dtypes(include=['int64', 'int32']).columns
X[int_cols] = X[int_cols].astype(float)

# পাইপলাইন থেকে Preprocessor এবং XGBoost মডেল আলাদা করা
preprocessor = pipeline.named_steps['preprocessor']
xgb_model = pipeline.named_steps['classifier']

# SHAP-এর জন্য ডেটা ট্রান্সফর্ম করা এবং ফিচারগুলোর নাম বের করা
X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()
xgb_model.get_booster().feature_names = list(feature_names)

print("[✓] Model and Data loaded successfully.\n")

# ---------------------------------------------------------
# মেথড ১: XGBoost Built-in Feature Importance
# ---------------------------------------------------------
print("📊 1. Generating XGBoost Feature Importance Plot...")
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='weight', title='Top 10 Features by Weight')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# মেথড ২: Explainable AI (SHAP Analysis)
# ---------------------------------------------------------
print("🧠 2. Generating SHAP Summary Plot (Logic Analysis)...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_transformed)

# SHAP সামারি প্লট
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
plt.title("SHAP Feature Impact on Churn Prediction")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# মেথড ৩: Partial Dependence Plot (PDP)
# ---------------------------------------------------------
print("📈 3. Generating Partial Dependence Plots...")
fig, ax = plt.subplots(figsize=(12, 5))
# আমরা সরাসরি মেইন পাইপলাইন ব্যবহার করব PDP এর জন্য
features_to_plot = ['tenure', 'MonthlyCharges']
display = PartialDependenceDisplay.from_estimator(
    pipeline, X, features_to_plot, kind='average', ax=ax
)
display.figure_.suptitle("How Tenure & Monthly Charges affect Churn Probability")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# মেথড ৪: MiniBatchKMeans Centroid Analysis
# ---------------------------------------------------------
print("\n🎯 4. Analyzing K-Means Cluster Centroids...")
numeric_features = df[['tenure', 'MonthlyCharges', 'TotalCharges']].values
clustering = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=20)
clustering.fit(numeric_features)

centroids = pd.DataFrame(
    clustering.cluster_centers_, 
    columns=['Avg Tenure', 'Avg MonthlyCharges', 'Avg TotalCharges'],
    index=['Cluster 0 (Budget / High Risk)', 'Cluster 1 (Core Stable)', 'Cluster 2 (High-Value)']
)
print("\n--- Cluster Characteristics ---")
print(centroids.round(2))
print("="*50)