# 🌊 Flood Forecasting Dashboard - Sylhet, Bangladesh

This is a Streamlit-based interactive dashboard for predicting flood-prone areas in the Sylhet region using satellite data and ensemble machine learning models.

## 👤 Author

**Developed by:** A climate researcher and urban planner at the Department of Environment, Bangladesh,  
Working on the NC4/BTR1 project and specialized in climate-resilient infrastructure and emission inventories.

---

## 📁 Files Included

- `flood_dashboard.py` – Main Streamlit app (upload → train → predict → visualize)
- `requirements.txt` – Python dependencies for deployment

## 🧠 Features

- Upload CSV files exported from Google Earth Engine (NDWI, Rainfall, Elevation, Flood)
- Train an ensemble model: Random Forest + XGBoost + Logistic Regression
- Interactive dashboard with:
  - Model Accuracy
  - Classification Report
  - Confusion Matrix

## 🛠️ Installation (Local)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/flood-dashboard.git
cd flood-dashboard

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run flood_dashboard.py
```

## 🚀 Deploy on Streamlit Cloud

1. Push your repo to GitHub.
2. Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Link your GitHub account and select this repository.
4. Set the main file as `flood_dashboard.py`.
5. Click **Deploy**.

## 📊 Input Data Format

CSV file must include these columns (exported from Google Earth Engine):

- `NDWI` (Normalized Difference Water Index)
- `Rainfall` (Total during a season)
- `Elevation` (from SRTM or other DEM)
- `Flood` (binary: 1 = flood-affected, 0 = dry)

## 📌 Use Case

Supports early flood prediction and planning in Bangladesh's haor wetlands and other vulnerable lowlands.  
Helps in mainstreaming geospatial machine learning into national climate resilience strategies.

---

🛰️ Powered by Earth Engine, Streamlit, Scikit-learn, and XGBoost.
