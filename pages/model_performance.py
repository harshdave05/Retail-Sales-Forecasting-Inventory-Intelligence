import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"

st.title("📊 Model Performance")
st.caption("Evaluation of the Random Forest sales forecasting model")

# --------------------------------------------------
# 2. LOAD EVALUATION FILES
# --------------------------------------------------
evaluation_file = TRAINING_DIR / "evaluation_results.csv"
comparison_file = TRAINING_DIR / "model_comparison.csv"
# --------------------------------------------------
# 3. OVERALL MODEL METRICS
# --------------------------------------------------
if evaluation_file.exists():
    results = pd.read_csv(evaluation_file)

    required_columns = {"Actual Sales", "Predicted Sales"}

    if required_columns.issubset(results.columns) and not results.empty:
        actual = results["Actual Sales"]
        predicted = results["Predicted Sales"]

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        st.header("Overall Evaluation")

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.3f}")
        col2.metric("RMSE", f"{rmse:.3f}")
        col3.metric("Test Samples", f"{len(results):,}")

        st.caption(
            "Lower MAE and RMSE generally indicate more accurate "
            "predictions. Results represent the selected product-store subset."
        )

        st.subheader("Actual vs Predicted Sales")

        chart_results = results.head(100).copy().reset_index(drop=True)
        chart_results.index = chart_results.index + 1
        chart_results.index.name = "Test Example"

        st.line_chart(
            chart_results[["Actual Sales", "Predicted Sales"]]
        )

        if "Product_ID" in results.columns:
            st.subheader("Evaluation Data")
            st.dataframe(
                results.head(100),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("Evaluation file is empty or has missing columns.")
else:
    st.warning(
        f"Evaluation file not found: {evaluation_file}\n\n"
        "Run the evaluation script from the project root using "
        "`python training/evaluate_model.py`."
    )

# --------------------------------------------------
# 4. RANDOM FOREST VS BASELINE
# --------------------------------------------------
st.divider()
st.header("Random Forest vs Baseline")

if comparison_file.exists():
    comparison = pd.read_csv(comparison_file)

    if {"Model", "MAE", "RMSE"}.issubset(comparison.columns):
        st.dataframe(
            comparison.round(3),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Error Comparison — Lower Is Better")
        st.bar_chart(comparison.set_index("Model")[["MAE", "RMSE"]])

        rf = comparison[comparison["Model"] == "Random Forest"]
        baseline = comparison[comparison["Model"] == "Baseline"]

        if not rf.empty and not baseline.empty:
            baseline_mae = baseline.iloc[0]["MAE"]

            if baseline_mae != 0:
                improvement = (
                    (baseline_mae - rf.iloc[0]["MAE"])
                    / baseline_mae * 100
                )
                st.write(
                    f"Random Forest MAE improvement over baseline: "
                    f"{improvement:.2f}%"
                )
    else:
        st.warning("Model comparison file has missing columns.")
else:
    st.info(
        "Model comparison file not found. Run "
        "`python training/evaluate_model.py`."
    )

# --------------------------------------------------
# 6. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Evaluation uses 200 selected product-store records from the M5 dataset. "
    "Results should not be interpreted as performance on the entire dataset."
)