import pandas as pd
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('C:/Users/HP/PycharmProjects/Machine_Learning/Autism Project/trained_model.sav', 'rb'))

# Load encoders
with open("C:/Users/HP/PycharmProjects/Machine_Learning/Autism Project/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


# Prediction function
def autism_prediction(input_data):
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for column in encoders:
        if column in input_df.columns:
            try:
                input_df[column] = encoders[column].transform(input_df[column])
            except ValueError as e:
                st.warning(f"Warning: Unseen value in '{column}': {input_df[column].values[0]}")
                return "Prediction could not be made due to unexpected input value."

    # Make prediction
    prediction = loaded_model.predict(input_df)

    return "Autism Spectrum Disorder (ASD) is predicted." if prediction[0] == 1 else "Autism Spectrum Disorder (ASD) is not predicted."


# Main app
def main():
    # Add background image using custom style
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url('https://images.unsplash.com/photo-1602525827871-95117eaa70ea?crop=entropy&cs=tinysrgb&fit=max&ixid=MnwzNjUyOXwwfDF8c2VhY2h8NXx8YXBzdHJhY3QlMjB3aXRoJTIwbmF0dXJlJTIwY29sb3J8ZW58MHx8fHwxNjg5NzAxMTMz&ixlib=rb-1.2.1&q=80&w=1080') no-repeat center center fixed;
            background-size: cover;
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
        }
        .stTitle {
            font-family: 'Arial', sans-serif;
            color: white;
        }
        .stTextInput, .stSelectbox, .stNumberInput {
            background-color: rgba(255, 255, 255, 0.7);
            color: black;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("🧠 Autism Prediction Web App")

    st.markdown("### Please enter the details below:")

    # Input fields with validation
    A1_Score = st.number_input("A1_Score (0 or 1) Social Skills:", min_value=0, max_value=1, step=1)
    A2_Score = st.number_input("A2_Score (0 or 1) Attention Switching:", min_value=0, max_value=1, step=1)
    A3_Score = st.number_input("A3_Score (0 or 1) Attention to Detail:", min_value=0, max_value=1, step=1)
    A4_Score = st.number_input("A4_Score (0 or 1) Communication:", min_value=0, max_value=1, step=1)
    A5_Score = st.number_input("A5_Score (0 or 1) Imagination:", min_value=0, max_value=1, step=1)
    A6_Score = st.number_input("A6_Score (0 or 1) Routine/Change Sensitivity:", min_value=0, max_value=1, step=1)
    A7_Score = st.number_input("A7_Score (0 or 1) Empathy:", min_value=0, max_value=1, step=1)
    A8_Score = st.number_input("A8_Score (0 or 1) Social Imagination:", min_value=0, max_value=1, step=1)
    A9_Score = st.number_input("A9_Score (0 or 1) Social Interaction:", min_value=0, max_value=1, step=1)
    A10_Score = st.number_input("A10_Score (0 or 1) Response to Social Situations:", min_value=0, max_value=1, step=1)

    Age = st.text_input("Age:")

    Gender = st.selectbox("Gender:", ["m", "f"])
    Ethnicity = st.text_input("Ethnicity (e.g., White-European, Black, Asian, etc.):")
    Jaundice = st.selectbox("Born with Jaundice?", ["yes", "no"])
    Autism = st.selectbox("Family member with Autism?", ["yes", "no"])
    country_of_res = st.text_input("Country of Residence:")
    used_app_before = st.selectbox("Used screening app before?", ["yes", "no"])
    result = st.text_input("Screening Score (Result):")
    relation = st.selectbox("Who is filling the form?", [
        "Self", "Others"
    ])

    diagnosis = ""

    if st.button("Get Prediction"):
        input_data = {
            'A1_Score': A1_Score,
            'A2_Score': A2_Score,
            'A3_Score': A3_Score,
            'A4_Score': A4_Score,
            'A5_Score': A5_Score,
            'A6_Score': A6_Score,
            'A7_Score': A7_Score,
            'A8_Score': A8_Score,
            'A9_Score': A9_Score,
            'A10_Score': A10_Score,
            'age': float(Age),
            'gender': Gender,
            'ethnicity': Ethnicity,
            'jaundice': Jaundice,
            'austim': Autism,  # typo retained to match model training
            'contry_of_res': country_of_res,  # typo retained to match model training
            'used_app_before': used_app_before,
            'result': float(result),
            'relation': relation
        }

        diagnosis = autism_prediction(input_data)
        st.success(diagnosis)


if __name__ == "__main__":
    main()
