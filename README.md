# AgriShield

AgriShield is a GPU-accelerated food security early warning system built during the NVIDIA x Presidency University AI GPU Summer Internship. It combines weather and consumer price index data with machine learning to forecast food price pressure and shortage risk several months ahead, helping identify vulnerable regions before disruptions rise.

## Project objective

Food systems are highly sensitive to climate variation, market conditions, and regional imbalances. AgriShield was designed to provide an early-warning layer for food security planning by modeling price and risk signals using historical and current data.

## What the system does

- Forecasts price movement and risk using data-driven modeling
- Uses weather and CPI signals to capture regional vulnerability
- Applies a GPU-accelerated ML pipeline for faster experimentation and iteration
- Highlights high-risk regions when probability thresholds are crossed
- Supports policy-oriented monitoring and decision support

## Tech stack

- Python
- Jupyter Notebooks
- pandas and NumPy
- scikit-learn
- NVIDIA RAPIDS (cuDF, cuML)
- matplotlib
- Weather and CPI API data sources

## Model approach

The project evaluates predictive models for both price forecasting and risk estimation, with a focus on performance and interpretability. Random forest models form a key part of the forecasting workflow, and the platform is designed to surface risk signals when local conditions start trending beyond monitored thresholds.

## Key results

The project reports strong predictive performance for the target forecasting task, including a high R² score on the evaluated dataset and a risk alerting mechanism triggered when regional conditions exceed a defined threshold.

## Project status

This is a research and prototype project focused on applied machine learning for food security analytics and operational monitoring.

## Running locally

1. Open the notebooks in the repository root.
2. Install the required Python dependencies.
3. Run the forecasting workflow or dashboard notebooks to reproduce the analysis.
