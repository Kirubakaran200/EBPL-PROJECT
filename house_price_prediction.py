import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. Load dataset (California Housing, no CSV required)
data = fetch_california_housing(as_frame=True)
df = data.frame
df['SalePrice'] = df['MedHouseVal'] * 100000  # Rescale price

# 2. Prepare data
X = df.drop(['MedHouseVal', 'SalePrice'], axis=1)
y = df['SalePrice']

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Define models
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# 6. Evaluate models
print("📊 Cross-validated RMSE scores:")
results = {}
for name, model in models.items():
    rmse_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    avg_rmse = rmse_scores.mean()
    results[name] = avg_rmse
    print(f"{name}: {avg_rmse:.2f}")

# 7. Train best model
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# 8. Predict and evaluate
y_pred = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n✅ Best Model: {best_model_name}")
print(f"📉 Test RMSE: {test_rmse:.2f}")

# 9. Plot predictions
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"{best_model_name} - Actual vs Predicted")
plt.tight_layout()
plt.show()
