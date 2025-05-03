import streamlit as st
import joblib
import numpy as np

# Load the saved model and LabelEncoder
model = joblib.load('random_forest_model.joblib')
le = joblib.load('label_encoder.pkl')

# Streamlit UI
st.title("Iris Flower Species Prediction")

# Input fields for user to enter flower features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0)

# Prediction button
if st.button("Predict"):
    # Prepare input data for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = model.predict(input_data)

    # Decode the prediction (convert back to original species)
    predicted_species = le.inverse_transform(prediction)

    # Display the result
    st.write(f"The predicted species is: {predicted_species[0]}")

# Download button for LabelEncoder
st.download_button(
    label="Download LabelEncoder",
    data=open('label_encoder.pkl', 'rb').read(),
    file_name='label_encoder.pkl',
    mime='application/octet-stream'
)
