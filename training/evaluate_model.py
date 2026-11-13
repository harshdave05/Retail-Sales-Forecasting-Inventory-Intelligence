# 1. Load the trained model
import joblib
import numpy as np
import pandas as pd

saved_data = joblib.load("models/product_model.pkl")

model = saved_data["model"]
product_history = saved_data["product_history"]
feature_names = saved_data["feature_names"]

X = []
y = []
product_ids = []

# 2. Create test data from the later 20% of each product history
for product_id, history in product_history.items():

    sales = np.array(history, dtype=float)

    split_day = int(len(sales) * 0.8)

    for i in range(max(7, split_day), len(sales)):
        X.append(sales[i-7:i])
        y.append(sales[i])
        product_ids.append(product_id)

# Convert test data into DataFrame and array
X_test = pd.DataFrame(X, columns=feature_names)
y_test = np.array(y)

# 3. Generate Random Forest predictions
predictions = model.predict(X_test)

# 4. Generate baseline predictions
baseline_predictions = X_test["day_7"].values

# 5. Calculate MAE and RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error

rf_mae = mean_absolute_error(y_test, predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, predictions))

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\n--- Model Comparison ---")
print("Products evaluated:", len(product_history))
print(f"Random Forest MAE: {rf_mae:.3f}")
print(f"Random Forest RMSE: {rf_rmse:.3f}")
print(f"Baseline MAE: {baseline_mae:.3f}")
print(f"Baseline RMSE: {baseline_rmse:.3f}")

# 6. Save model comparison
comparison = pd.DataFrame({
    "Model": ["Random Forest", "Baseline"],
    "MAE": [rf_mae, baseline_mae],
    "RMSE": [rf_rmse, baseline_rmse]
})

comparison.to_csv("training/model_comparison.csv", index=False)

print("Saved training/model_comparison.csv")

# 7. Save actual and predicted sales
results = pd.DataFrame({
    "Product_ID": product_ids,
    "Actual Sales": y_test,
    "Predicted Sales": predictions
})

results.to_csv("training/evaluation_results.csv", index=False)

print("Saved training/evaluation_results.csv")
print("\nEvaluation completed successfully!")