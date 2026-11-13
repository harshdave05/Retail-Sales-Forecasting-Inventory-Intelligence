
# 1. Load the sales dataset
import pandas as pd

file_path = "dataset/sales_train_validation.csv"
data = pd.read_csv(file_path)

# 2. Select four stores
selected_stores = ["CA_1", "CA_2", "TX_1", "WI_1"]

# Extract store ID from product ID
data["store_id"] = data["id"].apply(lambda x: "_".join(x.split("_")[-3:-1]))

# Keep only selected stores
data = data[data["store_id"].isin(selected_stores)]

# Select 50 products from each store
data = data.groupby("store_id", group_keys=False).sample(n=50, random_state=42).reset_index(drop=True)

print("Selected stores:", selected_stores)
print("Product-store records:", len(data))

# 3. Prepare product sales history
import numpy as np

day_columns = [col for col in data.columns if col.startswith("d_")]

X = []
y = []
target_days = []
product_history = {}

for _, row in data.iterrows():
    product_id = row["id"]
    sales = row[day_columns].astype(float).values

    product_history[product_id] = sales.tolist()

    # Previous 7 days are used to predict the next day
    for i in range(7, len(sales)):
        X.append(sales[i-7:i])
        y.append(sales[i])
        target_days.append(i)

# 4. Convert data into arrays and DataFrame
feature_names = [f"day_{i}" for i in range(1, 8)]

X = pd.DataFrame(X, columns=feature_names)
y = np.array(y)
target_days = np.array(target_days)

# 5. Split data into training and testing sets
split_day = int(len(day_columns) * 0.8)

train_mask = target_days < split_day
test_mask = target_days >= split_day

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

# 6. Train the Random Forest model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)

# 7. Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Model training completed!")
print("Products used:", len(product_history))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 8. Save the trained model
import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "product_history": product_history,
        "feature_names": feature_names
    },
    "models/product_model.pkl"
)

print("Model saved successfully!")
