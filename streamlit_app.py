import streamlit as st
import pandas as pd
import joblib 

st.title("Aplicación, que predice el tipo de mascota")
st.write("Streamlit Machine Learning App")

# Input bar 1
height = st.number_input("Enter Height")

weight = st.number_input("Enter Weight")

color_eye = st.selectbox("Select Eye Colout",("Brown", "Blue"))

# Display the entered name
if st.button("submit"):

    pet_model = joblib.load("pet_model.pkl")
    
    X = pd.DataFrame([[height, weight, color_eye]], columns = ["Height", "Weight", "Eye"])

    X = X.replace(["Brown", "Blue"], [1, 0])

    prediction = pet_model.predict(X)[0]

    st.text(f"This instance is a {prediction}")

