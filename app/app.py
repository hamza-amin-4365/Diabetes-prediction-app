import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Function to load model and scaler
def load_model():
    diabetes = pd.read_csv('diabetes.csv')
    X = diabetes.drop('Outcome', axis=1)
    Y = diabetes['Outcome']

    # Set column names if not present
    if X.columns[0] == 0:
        X.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age']

    # Create and train the Support Vector Classifier (SVC)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svc_model = SVC(kernel='linear')
    svc_model.fit(X_scaled, Y)

    return svc_model, scaler

# Function to make a prediction
def predict_diabetes(model, scaler, user_input):
    # Standardize the input features
    user_input_scaled = scaler.transform([user_input])

    # Make prediction
    prediction = model.predict(user_input_scaled)

    return prediction[0]

# Function to create the Streamlit app
def main():
    # Set page title and icon
    st.set_page_config(page_title="Diabetes Prediction App", page_icon=":bar_chart:")

    # Load model and scaler
    model, scaler = load_model()

    # Add a header with custom styling
    st.markdown("# Diabetes Prediction App")

    # Upload image and display it with improved styling
    img = Image.open("img.jpg")
    img = img.resize((600, 400))
    st.image(img, caption="Diabetes Image", use_column_width=True)

    # Get user input using Streamlit's number_input
    st.markdown("## Enter Patient Information:")
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, step=12)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, step=10)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, step=3)
    insulin = st.number_input("Insulin Level", min_value=0, step=10)
    bmi = st.number_input("BMI", min_value=0, step=3)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0, step=1)

    # Make prediction when the user clicks the "Predict" button
    if st.button("Predict", key="predict_button"):
        user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
                      diabetes_pedigree_function, age]

        prediction = predict_diabetes(model, scaler, user_input)

        # Display prediction with custom styling
        if prediction == 1:
            st.error("Prediction: Diabetes Positive")
        else:
            st.success("Prediction: Diabetes Negative")

if __name__ == "__main__":
    main()
