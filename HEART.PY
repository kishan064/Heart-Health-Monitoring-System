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

# Streamlit UI
st.title('Heart Health Monitoring System')

st.header("Enter Patient Data")

# Full form mapping for features
feature_fullforms = {
    "age": "Age of the patient",
    "gender": "Gender ",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dL)",
    "fbs": "Fasting Blood Sugar (> 120 mg/dL)",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression due to Exercise",
    "slope": "Slope of ST Segment",
    "ca": "Number of Major Vessels Colored",
    "thal": "Thalassemia"
}

# Input fields for user data
user_inputs = {}
invalid_input = False

for feature in X.columns:
    label = feature_fullforms.get(feature, feature)
    help_text = label

    if feature == "gender":
        gender = st.selectbox(f'{label}', ["Select Gender", "Male", "Female"], help="Select Gender: 1 = male, 0 = female")
        if gender == "Select Gender":
            invalid_input = True
        else:
            user_inputs[feature] = 1 if gender == "Male" else 0

    elif feature == "age":
        user_inputs[feature] = st.number_input(f'{label}', min_value=1, step=1, format="%d", help="Enter patient's age in years", placeholder="Enter Age")

    elif feature == "cp":
        cp_options = {
            "Select Chest Pain Type": -1,
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }
        cp_choice = st.selectbox(f'{label}', list(cp_options.keys()), help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
        user_inputs[feature] = cp_options[cp_choice]
        if cp_options[cp_choice] == -1:
            invalid_input = True

    elif feature == "fbs":
        fbs_options = {"Select Option": -1, "True": 1, "False": 0}
        fbs_choice = st.selectbox(f'{label}', list(fbs_options.keys()), help="1 = True, 0 = False")
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
        restecg_choice = st.selectbox(f'{label}', list(restecg_options.keys()), help="0: Normal, 1: ST-T abnormality, 2: Left ventricular hypertrophy")
        user_inputs[feature] = restecg_options[restecg_choice]
        if restecg_options[restecg_choice] == -1:
            invalid_input = True

    elif feature == "exang":
        exang_options = {"Select Option": -1, "Yes": 1, "No": 0}
        exang_choice = st.selectbox(f'{label}', list(exang_options.keys()), help="1 = Yes, 0 = No")
        user_inputs[feature] = exang_options[exang_choice]
        if exang_options[exang_choice] == -1:
            invalid_input = True

    elif feature == "slope":
        slope_options = {"Select ST Slope": -1, "Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope_choice = st.selectbox(f'{label}', list(slope_options.keys()), help="0: Upsloping, 1: Flat, 2: Downsloping")
        user_inputs[feature] = slope_options[slope_choice]
        if slope_options[slope_choice] == -1:
            invalid_input = True

    elif feature == "ca":
        user_inputs[feature] = st.selectbox(f'{label}', [0, 1, 2, 3], help="Range: 0-3")

    elif feature == "thal":
        thal_options = {"Select Thalassemia Type": -1, "Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
        thal_choice = st.selectbox(f'{label}', list(thal_options.keys()), help="1: Normal, 2: Fixed Defect, 3: Reversible Defect")
        user_inputs[feature] = thal_options[thal_choice]
        if thal_options[thal_choice] == -1:
            invalid_input = True

    elif feature == "trestbps":
        user_inputs[feature] = round(st.number_input(f'{label}', min_value=1.0, value=120.0, step=1.0, help="Range: Systolic: 70 – 300 mm Hg , Diastolic: 40 – 150 mm Hg", placeholder="e.g. 130"))

    elif feature == "chol":
        chol_value = st.number_input(f'{label}', min_value=1, format="%d", help="Typical Range: 126 - 564", placeholder="e.g. 200")
        if chol_value < 0:
            st.warning("Serum Cholesterol cannot be negative. Please enter a valid value.")
            invalid_input = True
        user_inputs[feature] = chol_value

    elif feature == "thalach":
        thalach_value = st.number_input(f'{label}', min_value=1, format="%d", help="Typical Range: 71 - 202", placeholder="e.g. 150")
        if thalach_value < 0:
            st.warning("Maximum Heart Rate Achieved cannot be negative. Please enter a valid value.")
            invalid_input = True
        user_inputs[feature] = thalach_value

    elif feature == "oldpeak":
        oldpeak_value = st.number_input(f'{label}', min_value=0.0, format="%f", help="Range: 0.0 - 6.2", placeholder="e.g. 1.4")
        if oldpeak_value < 0:
            st.warning("ST Depression Induced by Exercise Relative to Rest cannot be negative. Please enter a valid value.")
            invalid_input = True
        user_inputs[feature] = oldpeak_value

    else:
        user_inputs[feature] = st.number_input(f'{label}', value=0, step=1, format="%d", help=help_text, placeholder="e.g. 1")

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
# st.subheader("Model Performance")
# st.write(f"Training Data Accuracy: **{training_data_accuracy:.2f}**")
# st.write(f"Test Data Accuracy: **{test_data_accuracy:.2f}**")

# Dataset preview
# st.subheader("Dataset Preview")
# st.dataframe(heart_data.head())
