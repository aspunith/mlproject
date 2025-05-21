import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import r2_score

# Load the trained model
MODEL_PATH = 'artifacts/trained_model.pkl'
PREPROCESSOR_PATH = 'artifacts/preprocessor.pkl'

# Load the model and preprocessor
def load_model():
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(PREPROCESSOR_PATH, 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    return model, preprocessor

# Streamlit app
def main():
    st.title("Student Performance Prediction")
    st.write("Enter the details below to predict student performance.")

    # Load model and preprocessor
    model, preprocessor = load_model()

    # User input form
    with st.form("prediction_form"):
        st.write("### Input Features")

        # Categorical inputs
        gender = st.selectbox("Gender", ["male", "female"], index=0)
        race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"], index=0)
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            ["some high school", "high school", "associate's degree", "some college", "bachelor's degree", "master's degree"],
            index=0
        )
        lunch = st.selectbox("Lunch", ["standard", "free/reduced"], index=0)
        test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"], index=0)

        # Numerical inputs
        math_score = st.number_input("Math Score", min_value=0, max_value=100, step=1)
        reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
        writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create a DataFrame from user input
        user_data = pd.DataFrame({
            "gender": [gender],
            "race_ethnicity": [race_ethnicity],
            "parental_level_of_education": [parental_level_of_education],
            "lunch": [lunch],
            "test_preparation_course": [test_preparation_course],
            "math_score": [math_score],
            "reading_score": [reading_score],
            "writing_score": [writing_score]
        })

        # Preprocess the user input
        try:
            processed_data = preprocessor.transform(user_data)
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            return

        # Make predictions
        predictions = model.predict(processed_data)
        st.write("### Prediction Result")
        st.write(f"Predicted Score: {predictions[0]}")

        # Evaluate performance category
        if predictions[0] > 85:
            performance = "Outstanding"
        elif 65 < predictions[0] <= 85:
            performance = "Good"
        elif 50 < predictions[0] <= 65:
            performance = "Scope for Improvement"
        else:
            performance = "Below Average"

        st.write(f"Performance Category: {performance}")

if __name__ == "__main__":
    main()
