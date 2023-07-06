import streamlit as st
import pandas as pd
from PIL import Image
from src.logger import logging
from src.utils import load_object

preprocessor = load_object("artifacts/preprocessor.pkl")
model = load_object("artifacts/model.pkl")

def main():
    # Set the page title and custom CSS styles
    st.markdown("""
        <style>
            .title {
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 30px;
            }
            .prediction {
                font-size: 24px;
                color: #1f77b4;
                margin-top: 30px;
            }
            .context {
                font-size: 18px;
                margin-top: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display the page title
    st.markdown('<h1 class="title">Concrete Compressive Strength</h1>', unsafe_allow_html=True)

    # Create input fields for the user to enter parameter values
    cement = st.number_input("Cement", min_value=0.0, max_value=1000.0, help="Enter the amount of cement in kg")
    slag = st.number_input("Blast Furnace Slag", min_value=0.0, max_value=500.0, help="Enter the amount of blast furnace slag in kg")
    fly_ash = st.number_input("Fly Ash", min_value=0.0, max_value=500.0, help="Enter the amount of fly ash in kg")
    water = st.number_input("Water", min_value=0.0, max_value=500.0, help="Enter the amount of water in kg")
    superplasticizer = st.number_input("Superplasticizer", min_value=0.0, max_value=100.0, help="Enter the amount of superplasticizer in kg")
    coarse_aggregate = st.number_input("Coarse Aggregate", min_value=0.0, max_value=2000.0, help="Enter the amount of coarse aggregate in kg")
    fine_aggregate = st.number_input("Fine Aggregate", min_value=0.0, max_value=1000.0, help="Enter the amount of fine aggregate in kg")
    age = st.number_input("Age", min_value=0, max_value=1000, help="Enter the age of the concrete in days")

    # Create a button to trigger the prediction
    if st.button("Predict"):
        # Create a DataFrame with the input parameters
        input_data = pd.DataFrame({
            'Cement': [cement],
            'Blast Furnace Slag': [slag],
            'Fly Ash': [fly_ash],
            'Water': [water],
            'Superplasticizer': [superplasticizer],
            'Coarse Aggregate': [coarse_aggregate],
            'Fine Aggregate': [fine_aggregate],
            'Age': [age]
        })

        # Preprocess the input data using the preprocessor object
        processed_data = preprocessor.transform(input_data)

        # Make predictions using the model
        prediction = model.predict(processed_data)

        # Display the prediction to the user with context
        st.markdown('The Compressive strength of concrete : ' + repr(prediction[0]), unsafe_allow_html=True)
        st.balloons()


if __name__ == "__main__":
    main()
