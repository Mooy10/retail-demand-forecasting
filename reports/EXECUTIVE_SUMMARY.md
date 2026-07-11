# Executive Summary

## What Was Solved

This project built a professional retail demand planning workflow using the public M5 dataset. The goal was to forecast short-term demand, validate model choices honestly, and translate forecast results into simulated inventory planning recommendations for a retail business context.

## Data Used

The project uses historical M5 retail data: sales, calendar attributes, event metadata, SNAP indicators, and sell prices. The official forecasting layer uses 70 store-department series and a 28-day horizon.

## Methodology

The workflow includes data ingestion, data understanding, demand segmentation, baseline forecasting, machine learning, hybrid model selection, out-of-sample validation, empirical uncertainty estimation, and simulated inventory planning. The final dashboard reads processed artifacts and does not retrain models from the UI.

## Official Forecast

The official forecast is the validated hybrid selector output for a 28-day horizon. The holdout validation result was WAPE 11.54% and RMSSE 0.762. Fallback logic was used for 10 series to keep low-confidence or unavailable predictions transparent.

## Simulated Inventory Decisions

The project simulates safety stock, reorder points, order-up-to levels, order quantities, stockout risk, and cost impact. Under the base scenario, the simulation generated 56 recommended orders, estimated total cost of 1,680,537.514, and estimated savings of 28,643.443 versus the baseline policy.

## Results

- 70 store-department series forecasted.
- 28-day demand forecast: 1,165,892.25 units.
- Holdout WAPE: 11.54%.
- Holdout RMSSE: 0.762.
- 72 automated tests passed.
- Dashboard validated across all pages, filters, charts, tables, and downloads.

## Limitations

Inventory and financial outputs are simulated. The project does not include real inventory, supplier constraints, open purchase orders, logistics capacity, or true operating costs. It is a portfolio project using public data and is not an official Walmart solution.

## Recommendation

Use the project as a portfolio case study showing end-to-end forecasting, analytics engineering, validation discipline, dashboard communication, and business translation. The next step would be a compact executive PDF or a hosted demo with generated sample artifacts.
