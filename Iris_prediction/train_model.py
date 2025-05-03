import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
df = pd.read_csv("IRIS.csv")

# Encode species labels (target variable)
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Save the LabelEncoder to a file
joblib.dump(le, 'label_encoder.pkl')

# Split features (X) and target (y)
X = df.drop('species', axis=1)
y = df['species']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'random_forest_model.joblib')

print("Model and LabelEncoder saved successfully.")
