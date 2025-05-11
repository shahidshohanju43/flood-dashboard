import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Sylhet Flood Forecast", layout="wide")
st.title("Flood Forecasting Dashboard - Sylhet, Bangladesh")

uploaded_file = st.file_uploader("Upload your Google Earth Engine exported CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")

    if st.checkbox("Show raw data"):
        st.dataframe(data.head())

    feature_cols = ['NDWI', 'Rainfall', 'Elevation']
    target_col = 'Flood'
    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    lr = LogisticRegression(max_iter=1000)

    ensemble = VotingClassifier(estimators=[
        ('RF', rf), ('XGB', xgb), ('LR', lr)
    ], voting='soft')

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.metric(label="Accuracy", value=f"{accuracy:.2f}")

    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.subheader("Classification Report")
    st.dataframe(report)

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.success("Prediction complete.")
else:
    st.info("Please upload a CSV exported from GEE with NDWI, Rainfall, Elevation, and Flood columns.")
