# Retail Sales Forecasting & Inventory Intelligence System

A machine learning-based retail sales forecasting and inventory management system built using Python, Random Forest Regression, and Streamlit.

The project forecasts next-day product sales using historical sales data and provides inventory recommendations to help users make better stock management decisions.

## Features

### 1. Sales Forecasting
- Predicts next-day sales using historical daily sales.
- Uses the previous seven days of sales as input features.
- Supports selected stores and product-store combinations.
- Displays historical sales trends.

### 2. Model Performance Evaluation
- Evaluates Random Forest Regression.
- Compares the model against a baseline forecasting method.
- Displays Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Visualizes actual versus predicted sales.

### 3. Inventory Intelligence
- Calculates average daily demand.
- Calculates the Reorder Point (ROP).
- Estimates required stock.
- Suggests order quantities.
- Displays inventory status to support replenishment decisions.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Random Forest model and evaluation |
| Joblib | Model serialization |
| M5 Retail Dataset | Historical retail sales data |

## Project Structure

```text
Retail-Inventory-System/
│
├── dataset/
│   └── calendar.csv
│
├── models/
│   └── product_model.pkl       # Generated after model training
│
├── pages/
│   └── model_performance.py
│
├── training/
│   ├── train.py
│   ├── evaluate_model.py
│   ├── evaluation_results.csv
│   └── model_comparison.csv
│
├── app.py
├── requirements.txt
└── .gitignore
## Dataset Setup

This project uses data from the M5 Retail Forecasting dataset.

The large dataset files are excluded from this GitHub repository because of their file sizes.

To reproduce model training:

1. Obtain the `sales_train_validation.csv` file from the M5 dataset.
2. Create the `dataset` folder if it does not already exist.
3. Place the file inside the dataset folder:

```text
dataset/sales_train_validation.csv
```

The current training workflow uses this sales file. The trained model file is generated locally and is not included in the repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/harshdave05/Retail-Sales-Forecasting-Inventory-Intelligence.git
```

### 2. Open the project directory

```bash
cd Retail-Sales-Forecasting-Inventory-Intelligence
```

### 3. Create a virtual environment

```bash
python -m venv virtual
```

### 4. Activate the virtual environment

For Windows PowerShell:

```powershell
.\virtual\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Model Training

After placing the required dataset file in the dataset folder, run:

```bash
python training/train.py
```

This trains the Random Forest Regression model and saves the generated model in the `models` directory.

## Model Evaluation

Run the evaluation script:

```bash
python training/evaluate_model.py
```

The script compares Random Forest performance with the baseline method and generates evaluation result files in the `training` directory.

## Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Machine Learning Approach

The project uses Random Forest Regression for next-day retail sales forecasting.

- **Algorithm:** Random Forest Regressor
- **Number of estimators:** 100
- **Random state:** 42
- **Input features:** Previous seven days of sales
- **Prediction target:** Next-day sales

The current implementation selects four stores:

- CA_1
- CA_2
- TX_1
- WI_1

The training workflow uses 200 selected product-store records.

## Inventory Calculation

The inventory module uses the following calculations:

**Average Daily Demand**

Mean of the previous seven days of sales.

**Reorder Point (ROP)**

```text
ROP = Average Daily Demand × Lead Time + Safety Stock
```

**Required Stock**

```text
Required Stock = Average Daily Demand × Days to Cover + Safety Stock
```

**Suggested Order Quantity**

```text
Order Quantity = Maximum(0, Required Stock - Current Stock)
```

The dashboard uses these calculations to display whether inventory is sufficient or requires replenishment.

## Model Evaluation Results

The evaluation script reported the following results:

| Metric | Random Forest | Baseline |
|---|---:|---:|
| MAE | 0.971 | 1.056 |
| RMSE | 1.893 | 2.295 |

Lower MAE and RMSE values generally indicate better prediction accuracy.

These results relate to the selected evaluation subset and should not be interpreted as performance across the entire M5 dataset.

## Future Scope

- Support for additional stores and product categories.
- Integration of more retail demand factors.
- Advanced time-series forecasting models.
- Automated inventory alerts.
- More comprehensive inventory optimization.

## Author

**Harsh Dave**  
BE – Information Technology  
Chandigarh University

GitHub: [harshdave05](https://github.com/harshdave05)

## Project Repository

[Retail Sales Forecasting & Inventory Intelligence System](https://github.com/harshdave05/Retail-Sales-Forecasting-Inventory-Intelligence)
