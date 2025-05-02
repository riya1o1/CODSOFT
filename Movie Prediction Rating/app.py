import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load model
model = load_model("movie_rating_model.h5")

# Load dataset (needed for categories in encoder)
df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')
df['Actors'] = df['Actor 1'] + ', ' + df['Actor 2'] + ', ' + df['Actor 3']
df.dropna(subset=['Genre', 'Director', 'Actors', 'Rating'], inplace=True)

# Define features
features = ['Genre', 'Director', 'Actors']
X = df[features]

# Fit encoder
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), features)]
)
preprocessor.fit(X)

# Streamlit UI
st.title("🎬 Movie Rating Predictor")

genre = st.selectbox("Select Genre", sorted(df['Genre'].unique()))
director = st.selectbox("Select Director", sorted(df['Director'].unique()))
actor1 = st.selectbox("Select Actor 1", sorted(df['Actor 1'].unique()))
actor2 = st.selectbox("Select Actor 2", sorted(df['Actor 2'].unique()))
actor3 = st.selectbox("Select Actor 3", sorted(df['Actor 3'].unique()))

# Combine actors
actors = actor1 + ", " + actor2 + ", " + actor3

# Prepare input DataFrame
input_df = pd.DataFrame([[genre, director, actors]], columns=features)

# Encode input
X_input_encoded = preprocessor.transform(input_df)

# Predict
rating_pred = model.predict(X_input_encoded)
st.success(f"⭐ Predicted IMDb Rating: {round(rating_pred[0][0], 2)}")
