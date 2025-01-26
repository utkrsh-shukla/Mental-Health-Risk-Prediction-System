import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model and preprocessor
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')  # Save and load the preprocessor

# Streamlit app
st.title("Depression Prediction App")
st.write("This app predicts the likelihood of depression based on user input.")

# Input fields
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', 10, 100, 25)
    work_study_hours = st.sidebar.slider('Work/Study Hours', 0, 24, 8)
    sleep_duration = st.sidebar.selectbox('Sleep Duration', ['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', 'More than 8 hours'])
    cgpa = st.sidebar.slider('CGPA', 0.0, 10.0, 7.5)
    financial_stress = st.sidebar.slider('Financial Stress', 0.0, 10.0, 5.0)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
    city = st.sidebar.text_input('City', 'New York')
    working_professional_or_student = st.sidebar.selectbox('Working Professional or Student', ['Working Professional', 'Student'])
    profession = st.sidebar.text_input('Profession', 'Engineer')
    degree = st.sidebar.selectbox('Degree', ['Bachelor', 'Master', 'PhD', 'Other'])
    dietary_habits = st.sidebar.selectbox('Dietary Habits', ['Healthy', 'Average', 'Unhealthy'])
    academic_pressure = st.sidebar.slider('Academic Pressure', 0.0, 10.0, 5.0)
    work_pressure = st.sidebar.slider('Work Pressure', 0.0, 10.0, 5.0)
    study_satisfaction = st.sidebar.slider('Study Satisfaction', 0.0, 10.0, 5.0)

    # Age Group Calculation
    if age <= 18:
        age_group = 'Teen'
    elif 18 < age <= 30:
        age_group = 'Young Adult'
    elif 30 < age <= 45:
        age_group = 'Middle-Aged'
    else:
        age_group = 'Senior'

    data = {
        'Age': age,
        'Work/Study Hours': work_study_hours,
        'Sleep Duration': sleep_duration,
        'CGPA': cgpa,
        'Financial Stress': financial_stress,
        'Gender': gender,
        'City': city,
        'Working Professional or Student': working_professional_or_student,
        'Profession': profession,
        'Degree': degree,
        'Dietary Habits': dietary_habits,
        'Academic Pressure': academic_pressure,
        'Work Pressure': work_pressure,
        'Study Satisfaction': study_satisfaction,
        'Age_Group': age_group
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Add missing columns with default values
missing_columns = [
    'Name', 'Job Satisfaction', 'Have you ever had suicidal thoughts ?', 
    'Family History of Mental Illness', 'is_profession_missing'
]

for col in missing_columns:
    if col not in input_df.columns:
        input_df[col] = np.nan  # Fill with NaN or appropriate default values

# Handle Missing Values
input_df['Profession'] = input_df['Profession'].fillna(input_df['Working Professional or Student'])

# Map Sleep Duration to Numerical Values
sleep_mapping = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '6-7 hours': 3,
    '7-8 hours': 4,
    'More than 8 hours': 5
}
input_df['Sleep Duration'] = input_df['Sleep Duration'].map(sleep_mapping)

# Create 'is_profession_missing' Feature
input_df['is_profession_missing'] = input_df['Profession'].isnull().astype(int)

# Ensure the columns are in the correct order
expected_columns = [
    'Name', 'Gender', 'Age', 'City', 'Working Professional or Student', 'Profession',
    'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
    'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
    'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness', 'is_profession_missing', 'Age_Group'
]

input_df = input_df[expected_columns]

# Display the user input
st.subheader('User Input Features')
st.write(input_df)

# Predict
if st.button('Predict'):
    # Preprocess the input data
    input_df_preprocessed = input_df

    # Make prediction
    prediction = model.predict(input_df_preprocessed)
    prediction_proba = model.predict_proba(input_df_preprocessed)

    # Display results
    st.subheader('Prediction')
    depression_status = np.array(['No Depression', 'Depression'])
    st.write(depression_status[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba) 
