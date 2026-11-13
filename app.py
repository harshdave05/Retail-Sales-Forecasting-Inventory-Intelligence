import os
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Sales & Inventory Intelligence",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Retail Sales Forecasting & Inventory Intelligence")
st.caption("Product-wise sales forecasting powered by Random Forest")

# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------
MODEL_FILE = "models/product_model.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("Model file not found. Run `python training/train.py` from the project root first.")
    st.stop()

saved_data = joblib.load(MODEL_FILE)

model = saved_data["model"]
product_history = saved_data["product_history"]
feature_names = saved_data["feature_names"]

# --------------------------------------------------
# 3. STORE AND PRODUCT SELECTION
# --------------------------------------------------
st.header("1. Sales Forecast")

# Group products according to their store
products_by_store = {}

for product_id in product_history.keys():
    parts = product_id.split("_")
    store = "_".join(parts[-3:-1])

    products_by_store.setdefault(store, []).append(product_id)

# Store selection
col1, col2 = st.columns(2)

with col1:
    selected_store = st.selectbox(
        "Select Store",
        sorted(products_by_store.keys())
    )

# Show products belonging only to the selected store
store_products = sorted(products_by_store[selected_store])

with col2:
    product_id = st.selectbox(
        "Select Product",
        store_products,
        format_func=lambda pid: "_".join(pid.split("_")[:3])
    )

# --------------------------------------------------
# 4. PRODUCT DETAILS AND SALES FORECAST
# --------------------------------------------------
parts = product_id.split("_")

category = parts[0] if len(parts) > 0 else "Unknown"
department = "_".join(parts[:2]) if len(parts) >= 2 else "Unknown"
item_id = "_".join(parts[:3]) if len(parts) >= 3 else "Unknown"
store = "_".join(parts[-3:-1])

col1, col2, col3 = st.columns(3)

col1.metric("Category", category)
col2.metric("Department", department)
col3.metric("Store", store)

st.caption(f"Item ID: {item_id}")

history = product_history[product_id]
last_7_days = history[-7:]

input_data = pd.DataFrame(
    [last_7_days],
    columns=feature_names
)

prediction = max(
    0.0,
    float(model.predict(input_data)[0])
)

forecast = math.ceil(prediction)

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Next-Day Sales", f"{forecast} units")

with col2:
    st.metric(
        "Average Sales (Last 7 Days)",
        f"{np.mean(last_7_days):.2f} units"
    )

st.caption(
    "Forecast uses the selected record's last seven historical sales "
    "values. It is not a live-data forecast."
)

chart_data = pd.DataFrame({
    "Day": [f"Day {i}" for i in range(1, 8)],
    "Sales": last_7_days
})

st.subheader("Historical Sales Trend")
st.line_chart(chart_data, x="Day", y="Sales")

# --------------------------------------------------
# 5. INVENTORY INTELLIGENCE
# --------------------------------------------------
st.divider()
st.header("2. Inventory Intelligence")

col1, col2 = st.columns(2)

with col1:
    current_stock = st.number_input(
        "Current stock (units)",
        min_value=0,
        value=20,
        step=1
    )

    lead_time = st.number_input(
        "Supplier lead time (days)",
        min_value=1,
        value=3,
        step=1
    )

with col2:
    safety_stock = st.number_input(
        "Safety stock (units)",
        min_value=0,
        value=5,
        step=1
    )

    days_to_cover = st.number_input(
        "Target stock coverage (days)",
        min_value=1,
        value=7,
        step=1
    )

average_daily_demand = float(np.mean(last_7_days))

reorder_point = math.ceil(
    average_daily_demand * lead_time + safety_stock
)

required_stock = math.ceil(
    average_daily_demand * days_to_cover + safety_stock
)

order_quantity = max(0, required_stock - current_stock)

col1, col2, col3 = st.columns(3)

col1.metric("Reorder Point", f"{reorder_point} units")
col2.metric("Required Stock", f"{required_stock} units")
col3.metric("Suggested Order", f"{order_quantity} units")

# Stock status
if current_stock <= reorder_point:
    st.error("Stock Status: REORDER REQUIRED")
    st.write(
        f"Your current stock is {current_stock} units, "
        f"and the reorder point is {reorder_point} units."
    )

elif current_stock < required_stock:
    st.warning("Stock Status: BELOW TARGET")
    st.write(
        f"You have {current_stock} units, but your target "
        f"stock requirement is {required_stock} units."
    )

else:
    st.success("Stock Status: SUFFICIENT")
    st.write(
        "Your current stock meets the estimated requirement."
    )

st.caption(
    "Reorder point = average daily demand × lead time + safety stock. "
    "Required stock = average daily demand × coverage days + safety stock. "
    "These estimates assume demand stays constant."
)

st.write("### Why this recommendation?")

st.write(
    f"- Average daily demand: {average_daily_demand:.2f} units"
)

st.write(
    f"- Expected demand during lead time: "
    f"{average_daily_demand * lead_time:.2f} units"
)

st.write(f"- Safety stock buffer: {safety_stock} units")

if current_stock <= reorder_point:
    st.write(
        f"- Current stock ({current_stock}) has reached "
        f"the reorder point ({reorder_point})."
    )
else:
    st.write(
        f"- Current stock ({current_stock}) is above "
        f"the reorder point ({reorder_point})."
    )

# --------------------------------------------------
# 6. FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Project note: Forecasting uses a selected subset of product-store "
    "records from the M5 dataset. Results may not generalize to all "
    "products or future retail conditions."
)