import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Set the page configuration at the very start
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")


# Load and preprocess the data
@st.cache_data
def load_model():
    df = pd.read_csv("Titanic-Dataset.csv")
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    # Fit the label encoder on the training data
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_sex, le_embarked


model, le_sex, le_embarked = load_model()

# Streamlit UI with improved layout and styling
st.title("🚢 Titanic Survival Prediction App")
st.markdown("### Enter passenger details below to predict if they survived or not.")

# Create columns for input fields to improve layout
col1, col2 = st.columns(2)

# Input fields in columns
with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3],
                          help="Select the passenger class (1 = Upper, 2 = Middle, 3 = Lower)")
    age = st.slider("Age", 0, 100, 25, help="Select the age of the passenger.")
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0, help="Number of siblings or spouses aboard.")
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0, help="Number of parents or children aboard.")

with col2:
    sex = st.selectbox("Sex", ['male', 'female'], help="Select the gender of the passenger.")
    fare = st.number_input("Fare Paid", 0.0, 600.0, 30.0, help="Select the fare paid by the passenger.")
    embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'],
                            help="Select the port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)")

# Encode inputs with validation
if sex not in ['male', 'female']:
    st.error("Invalid input for 'Sex'. Please select 'male' or 'female'.")
else:
    sex_encoded = le_sex.transform([sex])[0]

if embarked not in ['C', 'Q', 'S']:
    st.error("Invalid input for 'Embarked'. Please select 'C', 'Q', or 'S'.")
else:
    embarked_encoded = le_embarked.transform([embarked])[0]

# Predict when the button is pressed
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Survived 🎉" if prediction == 1 else "Did not survive 😔"
    st.subheader(f"**Prediction:** {result}")
    st.write("### Explanation:")
    st.write(f"Based on the data you provided, the passenger was predicted to {result.lower()}.")
    st.write("The model uses several factors such as age, gender, class, and others to make its prediction.")

    # Display more insights (optional)
    st.write("### Additional Insights")
    st.write(
        "This model is a random forest classifier trained on historical Titanic data, making predictions based on patterns in the data.")

# Add a footer with additional details
st.markdown("---")
st.markdown("#### Made with ❤️ by Riya Singh - Titanic Survival Prediction")
st.markdown("This is a basic machine learning model built with Streamlit.")
