# Case Study: Retail Planning & Forecasting Analytics

## 1. Executive Summary

This project turns the public M5 retail dataset into a professional demand planning case study. The final solution forecasts 28 days of demand for 70 store-department series, validates model choices with temporal holdout testing, and converts the forecast into simulated inventory recommendations. The goal is not to win a Kaggle leaderboard, but to show how a data scientist and analytics engineer can build a business-ready planning workflow.

## 2. Business Context

Retail teams need to decide what to buy, where to allocate stock, and how to respond to demand uncertainty. If forecasts are too low, stores risk stockouts and lost sales. If forecasts are too high, inventory ties up cash and increases holding cost. A useful forecasting project must connect model performance to operational decisions.

## 3. Dataset

The project uses the M5 Forecasting Accuracy dataset from Kaggle. It contains historical unit sales, calendar attributes, event metadata, SNAP indicators, and sell prices for Walmart retail data. The project aggregates the data to store-department level for the first official forecasting layer.

## 4. Data Challenges

The raw sales table is wide and large. Melting every item-store-day record would create tens of millions of rows. The project therefore uses memory-conscious aggregation, compact Parquet outputs, and focused modeling tables. Demand is also sparse and seasonal, which makes simple error metrics such as MAPE less reliable.

## 5. Demand Segmentation

Demand series were profiled with ADI and CV2 to identify smooth, erratic, intermittent, and lumpy patterns. ABC segmentation was used to understand volume concentration, and XYZ segmentation was used to classify variability. These segments helped explain why not every product behaves like a clean textbook time series.

## 6. Forecasting Strategy

The project uses temporal validation instead of random splits. The official horizon is 28 days. The first official planning layer is 70 store-department series, which balances business interpretability with computational feasibility.

## 7. Baselines

Baselines were built before advanced models. `seasonal_naive_28` remained a strong benchmark and is kept visible throughout the project. This is important because a professional forecasting workflow should not assume machine learning is automatically better than simple seasonal rules.

## 8. Machine Learning

Machine learning models included scikit-learn, XGBoost, and LightGBM approaches. The models used temporal, lag, rolling, calendar, event, and price-related features where appropriate. Training was intentionally constrained in some phases to keep the project runnable on a local workstation.

## 9. Why ML Did Not Always Win

Retail demand can be noisy, sparse, seasonal, and heavily affected by hierarchy level. Some ML models improved specific metrics such as RMSSE, but did not consistently dominate the seasonal baseline by WAPE. The project treats this as a real business finding rather than a failure.

## 10. Model Selection

A hybrid selector was built to choose the best available model per series. It uses prior validation evidence and includes confidence and fallback logic. This makes the final forecast more transparent than a single global model forced across all series.

## 11. Out-of-Sample Validation

The key validation step is a strict holdout window not used for model selection. The official hybrid forecast achieved WAPE 11.54% and RMSSE 0.762 on the W3 holdout, outperforming the required seasonal benchmark in the final validation setup.

## 12. Inventory Simulation

The validated forecast was translated into simulated inventory planning metrics: safety stock, reorder point, order-up-to level, recommended order quantity, projected stockouts, and estimated costs. These values are scenario-based simulations, not real Walmart inventory or financial data.

## 13. Dashboard

A Streamlit dashboard presents the project as an executive decision-support tool. It includes pages for overview, forecast detail, inventory planning, model performance, business insights, and methodology. It reads processed artifacts only and does not retrain models.

## 14. Business Insights

The final dashboard highlights priority stores and departments, fallback usage, confidence distribution, simulated inventory risk, and recommended actions. The most valuable insight is methodological: keep the benchmark visible, validate out of sample, and label simulated financial impact clearly.

## 15. Limitations

The project uses public historical data. It does not include real inventory, supplier lead times, open orders, capacity, substitution effects, or true replenishment constraints. The official forecast is at store-department level, not the full item-store level.

## 16. Lessons Learned

A professional forecasting project is not only about model accuracy. It also requires data contracts, validation, reproducibility, business framing, uncertainty communication, and a clear line between measured results and simulated impact.

## 17. Next Steps

Future work could extend the official forecast to prioritized item-store series, integrate real inventory data, add service-level optimization, deploy the dashboard, and create a PDF executive report for stakeholders.
