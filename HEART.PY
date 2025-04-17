import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import streamlit as st
import base64


# Load the dataset
heart_data = pd.read_csv('heart_disease_data.csv')

# Rename 'sex' column to 'gender' for clarity
heart_data.rename(columns={'sex': 'gender'}, inplace=True)

# Split features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Cache the trained model
@st.cache_resource
def train_model():
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model

model = train_model()

# Model accuracy
training_data_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_data_accuracy = accuracy_score(model.predict(X_test), Y_test)

# Streamlit UI
st.title('Heart Health Monitoring System')
# Set background image using base64 encoding
def set_background(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background image
set_background("heart_img3.webp")


st.header("Enter Patient Data")

# Full form mapping for features
feature_fullforms = {
    "age": "Age of the patient",
    "gender": "Gender (1 = male; 0 = female)",
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

# Input fields for user data
user_inputs = {}
invalid_input = False

for feature in X.columns:
    label = feature_fullforms.get(feature, feature)

    if feature == "gender":
        gender = st.selectbox(f'{label}', ["Select Gender", "Male", "Female"])
        if gender == "Select Gender":
            invalid_input = True
        else:
            user_inputs[feature] = 1 if gender == "Male" else 0

    elif feature == "age":
        user_inputs[feature] = st.number_input(f'{label}', min_value=1, step=1, format="%d")

    elif feature == "cp":
        cp_options = {
            "Select Chest Pain Type": -1,
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }
        cp_choice = st.selectbox(f'{label}', list(cp_options.keys()))
        user_inputs[feature] = cp_options[cp_choice]
        if cp_options[cp_choice] == -1:
            invalid_input = True

    elif feature == "fbs":
        fbs_options = {"Select Option": -1, "True": 1, "False": 0}
        fbs_choice = st.selectbox(f'{label}', list(fbs_options.keys()))
        user_inputs[feature] = fbs_options[fbs_choice]
        if fbs_options[fbs_choice] == -1:
            invalid_input = True

    elif feature == "restecg":
        restecg_options = {
            "Select ECG Result": -1,
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Left ventricular hypertrophy": 2
        }
        restecg_choice = st.selectbox(f'{label}', list(restecg_options.keys()))
        user_inputs[feature] = restecg_options[restecg_choice]
        if restecg_options[restecg_choice] == -1:
            invalid_input = True

    elif feature == "exang":
        exang_options = {"Select Option": -1, "Yes": 1, "No": 0}
        exang_choice = st.selectbox(f'{label}', list(exang_options.keys()))
        user_inputs[feature] = exang_options[exang_choice]
        if exang_options[exang_choice] == -1:
            invalid_input = True

    elif feature == "slope":
        slope_options = {"Select ST Slope": -1, "Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope_choice = st.selectbox(f'{label}', list(slope_options.keys()))
        user_inputs[feature] = slope_options[slope_choice]
        if slope_options[slope_choice] == -1:
            invalid_input = True

    elif feature == "ca":
        user_inputs[feature] = st.selectbox(f'{label}', [0, 1, 2, 3])

    elif feature == "thal":
        thal_options = {"Select Thalassemia Type": -1, "Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
        thal_choice = st.selectbox(f'{label}', list(thal_options.keys()))
        user_inputs[feature] = thal_options[thal_choice]
        if thal_options[thal_choice] == -1:
            invalid_input = True

    elif feature == "trestbps":
        user_inputs[feature] = round(st.number_input(f'{label}', value=120.0, step=1.0))

    else:
        user_inputs[feature] = st.number_input(f'{label}', value=0, step=1, format="%d")

# Predict button
if st.button('Predict Heart Disease'):
    if invalid_input:
        st.warning("Please fill in all fields correctly before prediction.")
    else:
        input_array = np.array([[user_inputs[col] for col in X.columns]])
        prediction = model.predict(input_array)

        if prediction[0] == 0:
            st.success("This person does **not** have heart disease.")
        else:
            st.error("This person **has** heart disease.")

# Model performance display
st.subheader("Model Performance")
st.write(f"Training Data Accuracy: **{training_data_accuracy:.2f}**")
st.write(f"Test Data Accuracy: **{test_data_accuracy:.2f}**")

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(heart_data.head())
