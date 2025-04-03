import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import streamlit as st
from PIL import Image

# Load the dataset
heart_data = pd.read_csv('heart_disease_data.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model accuracy
training_data_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_data_accuracy = accuracy_score(model.predict(X_test), Y_test)

# Streamlit UI
st.title('Heart Health Monitoring System')
st.image('heart_img.jpg', width=450)

st.header("Enter Patient Data")

# Full form mapping for features
feature_fullforms = {
    "age": "Age of the patient",
    "gender": "Gender",
    "cp": "Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dL)",
    "fbs": "Fasting Blood Sugar (> 120 mg/dL, 1 = True, 0 = False)",
    "restecg": "Resting Electrocardiographic Results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1 = Yes, 0 = No)",
    "oldpeak": "ST Depression Induced by Exercise Relative to Rest",
    "slope": "Slope of the Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)",
    "ca": "Number of Major Vessels Colored by Fluoroscopy (0-3)",
    "thal": "Thalassemia Type (1: Normal, 2: Fixed Defect, 3: Reversible Defect)"
}

# Rename 'sex' column to 'gender'
X.rename(columns={'sex': 'gender'}, inplace=True)

# Generate input fields dynamically
user_inputs = {}
for feature in X.columns:
    label = feature_fullforms.get(feature, feature)
    if feature == "gender":
        user_inputs[feature] = st.selectbox(f'{label}', ["Male", "Female"])
    else:
        user_inputs[feature] = st.number_input(f'{label}', value=0, format="%d")

if st.button('Predict Heart Disease'):
    try:
        input_values = list(user_inputs.values())
        input_values[X.columns.get_loc("gender")] = 1 if input_values[X.columns.get_loc("gender")] == "Male" else 0
        input_data = np.array([input_values], dtype=int)
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.success("This person does not have heart disease.")
        else:
            st.error("This person has heart disease.")
    except ValueError:
        st.warning('Invalid input values, please check again.')

# Display Model Performance
st.subheader("Model Performance")
st.write(f"Training Data Accuracy: {training_data_accuracy:.2f}")
st.write(f"Test Data Accuracy: {test_data_accuracy:.2f}")

# Show dataset preview
st.subheader("Dataset Preview")
st.write(heart_data.head())
