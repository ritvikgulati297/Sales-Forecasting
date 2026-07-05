# Sales Forecasting & Demand Intelligence System

End to end sales forecasting and demand intelligence project built on the Superstore Sales dataset. Covers time series decomposition, three forecasting models compared head to head, anomaly detection, product demand clustering, and an interactive Streamlit dashboard.

**Author:** Ritvik Gulati

---

## What this project does

1. Loads and explores four years of Superstore sales data, extracts time features, and answers core business questions (top category, most consistent region, shipping delay, seasonality).
2. Decomposes monthly sales into trend, seasonal, and residual components, and tests stationarity with the Augmented Dickey Fuller test.
3. Builds and compares three forecasting models: SARIMA, Facebook Prophet, and XGBoost, on a real held out test period, then picks a winner using actual error metrics rather than preference.
4. Applies the best model to five business segments (Furniture, Technology, Office Supplies, West region, East region).
5. Detects anomalous sales weeks using two independent methods, Isolation Forest and rolling Z score, and compares where they agree and disagree.
6. Segments 17 product sub categories into four demand behaviors using K Means clustering and PCA, with a stocking recommendation per segment.
7. Ships a four page Streamlit dashboard covering overview, forecast explorer, anomaly report, and demand segments.
8. Summarizes everything in a two page executive report written for a non technical business audience.

## Repository structure

```
SalesForecasting_RitvikGulati/
  analysis.ipynb        Full notebook, Tasks 1 through 6, already executed with real outputs
  app.py                 Streamlit dashboard, Task 7
  train.csv               Superstore Sales dataset
  vgsales.csv            Supplementary Video Game Sales dataset
  summary.docx          Two page executive business report, Task 8
  requirements.txt      Pinned dependency versions
  charts/                  All chart images referenced in the notebook and report
```

## Setup

```bash
pip install -r requirements.txt
```

Prophet's install can be slow depending on your machine, since it compiles a Stan backend. Give it a few minutes on first install.

## Running the notebook

Open `analysis.ipynb` in Jupyter or Google Colab and run all cells top to bottom. It expects `train.csv` and `vgsales.csv` to sit next to it in the same folder.

## Running the dashboard locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The forecast page fits a SARIMA model live for whichever category or region you pick, results are cached so repeated selections are instant.

## Deploying the dashboard

1. Push this folder to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and point it at the repo and `app.py`.
3. Streamlit Community Cloud installs `requirements.txt` automatically and gives you a live link.

## Key results

- Technology is the highest revenue category ($827,456 over four years), and the only major category the model projects to keep growing.
- SARIMA was selected as the production model based on lowest RMSE against three real held out months, though XGBoost had a lower MAE and MAPE, both are documented honestly in the comparison table rather than only reporting the winner's numbers.
- Two anomaly detection methods agree on the strongest sales spikes and disagree on smaller ones, both results are shown rather than only the ones that agree.
- Product sub categories split into four demand groups: High Volume Stable, Growing, Declining, and Low Volume Volatile, each with a different recommended stocking approach.

## Known limitations

- **The video game sales dataset does not merge with the Superstore dataset.** It has no shared key, no compatible date granularity, and no product or region overlap. Rather than forcing a fake join, it is used as an independent second anomaly detection exercise, which is documented directly in the notebook.
- **Forecasts are at the company and segment level, not individual SKU level.** The confidence ranges are genuinely wide, roughly double between the low and high end some months, they should guide direction and safety stock sizing, not exact order quantities.
- The dashboard's forecast page refits SARIMA on demand for whichever segment is selected. On a live deployment, the first load of each new segment will take a few seconds while it fits.
