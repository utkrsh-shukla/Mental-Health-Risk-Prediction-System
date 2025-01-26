import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('model.pkl')
    return model

model = load_model()

# Define the sleep duration mapping
sleep_mapping = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '6-7 hours': 3,
    '7-8 hours': 4,
    'More than 8 hours': 5
}

# Streamlit App
st.title("Depression Analysis App")

# Input fields for prediction
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", min_value=0, max_value=100, value=25)
work_study_hours = st.sidebar.slider("Work/Study Hours", min_value=0, max_value=24, value=8)
sleep_duration = st.sidebar.selectbox("Sleep Duration", options=[1, 2, 3, 4, 5], format_func=lambda x: list(sleep_mapping.keys())[list(sleep_mapping.values()).index(x)])
cgpa = st.sidebar.slider("CGPA", min_value=0.0, max_value=4.0, value=3.0)
financial_stress = st.sidebar.slider("Financial Stress", min_value=0.0, max_value=10.0, value=5.0)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
city = st.sidebar.text_input("City", "New York")
working_professional_or_student = st.sidebar.selectbox("Working Professional or Student", options=["Working Professional", "Student"])
profession = st.sidebar.text_input("Profession", "Engineer")
degree = st.sidebar.selectbox("Degree", options=["Bachelor", "Master", "PhD"])
dietary_habits = st.sidebar.selectbox("Dietary Habits", options=["Healthy", "Unhealthy"])
academic_pressure = st.sidebar.slider("Academic Pressure", min_value=0.0, max_value=10.0, value=5.0)
work_pressure = st.sidebar.slider("Work Pressure", min_value=0.0, max_value=10.0, value=5.0)
study_satisfaction = st.sidebar.slider("Study Satisfaction", min_value=0.0, max_value=10.0, value=5.0)
age_group = st.sidebar.selectbox("Age Group", options=["Teen", "Young Adult", "Middle-Aged", "Senior"])

# Create a DataFrame from the input features
input_data = pd.DataFrame({
    'Age': [age],
    'Work/Study Hours': [work_study_hours],
    'Sleep Duration': [sleep_duration],
    'CGPA': [cgpa],
    'Financial Stress': [financial_stress],
    'Gender': [gender],
    'City': [city],
    'Working Professional or Student': [working_professional_or_student],
    'Profession': [profession],
    'Degree': [degree],
    'Dietary Habits': [dietary_habits],
    'Academic Pressure': [academic_pressure],
    'Work Pressure': [work_pressure],
    'Study Satisfaction': [study_satisfaction],
    'Age_Group': [age_group]
})

# Preprocess the input data
categorical_features = ['Gender', 'City', 'Working Professional or Student', 'Profession', 'Degree', 
                        'Dietary Habits', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Age_Group']
numerical_features = ['Age', 'Work/Study Hours', 'Sleep Duration', 'CGPA', 'Financial Stress']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the input data
input_data_preprocessed = preprocessor.fit_transform(input_data)

# Make a prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data_preprocessed)
    st.write(f"### Prediction: {'Depressed' if prediction[0] == 1 else 'Not Depressed'}")
