import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('logistic_model.pkl')

# Page title
st.title("💳 Credit Card Fraud Detection")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file with transaction(s)", type="csv")

if uploaded_file is not None:
    # Load CSV
    data = pd.read_csv(uploaded_file)

    # Check for required features
    if 'Amount' in data.columns and 'Time' in data.columns:
        # Preprocessing
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
        data.drop(['Time', 'Amount'], axis=1, inplace=True)

        # Predict
        predictions = model.predict(data)
        data['Prediction'] = predictions
        data['Prediction'] = data['Prediction'].map({0: 'Genuine', 1: 'Fraud'})

        # Display Results
        st.write("Prediction Results:")
        st.dataframe(data[['Prediction']])

        fraud_count = (predictions == 1).sum()
        st.success(f"Detected {fraud_count} fraudulent transaction(s).")
    else:
        st.error("CSV must contain 'Time' and 'Amount' columns for processing.")
