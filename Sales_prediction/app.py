# advertising_ui.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Advertising Sales Predictor", layout="wide")

st.title("📈 Advertising Sales Prediction using Linear Regression")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    with st.expander("🔍 Dataset Info"):
        st.write(df.info())

    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

    # Visualizations
    st.subheader("📉 Feature Correlation Heatmap")
    fig1, ax1 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax1)
    st.pyplot(fig1)

    st.subheader("🔗 Pairplot")
    st.text("This might take a few seconds...")
    fig2 = sns.pairplot(df)
    st.pyplot(fig2)

    # Feature and Target selection
    if all(col in df.columns for col in ['TV', 'Radio', 'Newspaper', 'Sales']):
        X = df[['TV', 'Radio', 'Newspaper']]
        y = df['Sales']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        st.subheader("✅ Model Evaluation Metrics")
        st.write(f"**Mean Absolute Error:** {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"**Root Mean Squared Error:** {mean_squared_error(y_test, y_pred, squared=False):.2f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

        # Actual vs Predicted
        st.subheader("🔍 Actual vs Predicted Sales")
        fig3, ax3 = plt.subplots()
        ax3.scatter(y_test, y_pred, color='blue')
        ax3.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
        ax3.set_xlabel("Actual Sales")
        ax3.set_ylabel("Predicted Sales")
        ax3.set_title("Actual vs Predicted Sales")
        ax3.grid(True)
        st.pyplot(fig3)
    else:
        st.warning("Required columns ['TV', 'Radio', 'Newspaper', 'Sales'] not found in uploaded file.")
else:
    st.info("Please upload a CSV file to get started.")
