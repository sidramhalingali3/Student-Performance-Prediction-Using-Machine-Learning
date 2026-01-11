import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")

st.title("Student Exam Score Predictor")

study_hours = st.number_input("Study Hours per Day", 0.0, 8.0, 0.0)
attendance = st.number_input("Attendance Percentage", 0,100,0)
sleep_hours = st.number_input("Sleep Hours per Night",0.0,8.0,0.0)

mental_health = 10
part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
tj_encoded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Exam Score"):
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, tj_encoded]])
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))
    st.success(f"Predicted Exam Percentage: {prediction:.2f}%")
