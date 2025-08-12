import pickle
import pandas as pd

# loading the saved model
loaded_model = pickle.load(open('C:/Users/HP/PycharmProjects/Machine_Learning/Autism Project/trained_model.sav', 'rb'))

# Input data
input_data = {
    "A1_Score": 1,
    "A2_Score": 1,
    "A3_Score": 0,
    "A4_Score": 0,
    "A5_Score": 1,
    "A6_Score": 0,
    "A7_Score": 1,
    "A8_Score": 0,
    "A9_Score": 0,
    "A10_Score": 0,
    "age": 25,
    "gender": "m",
    "ethnicity": "White-European",
    "jaundice": "no",
    "austim": "no",
    "contry_of_res": "United States",
    "used_app_before": "no",
    "result": 6,
    "relation": "Self"
}

# Convert the input_data dictionary to a pandas DataFrame
input_df = pd.DataFrame([input_data])

# Load the encoders
with open("C:/Users/HP/PycharmProjects/Machine_Learning/Autism Project/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Fill missing categorical columns with a default value (e.g., 'Unknown')
for column in encoders:
    if column not in input_df.columns:
        input_df[column] = "Unknown"

# Encode categorical features
for column in encoders:
    if column in input_df.columns:
        try:
            # Ensure each value is in list format for encoding
            input_df[column] = input_df[column].astype(str)
            input_df[column] = encoders[column].transform(input_df[column])
        except ValueError:
            # If unseen value, use the most frequent class from the encoder
            most_frequent = encoders[column].classes_[0]
            input_df[column] = most_frequent
            input_df[column] = encoders[column].transform(input_df[column])

# Load the expected column order and align the input_df
with open("C:/Users/HP/PycharmProjects/Machine_Learning/Autism Project/feature_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)

# Add missing columns with default value 0
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training data
input_df = input_df[expected_columns]

# Predict autism
prediction = loaded_model.predict(input_df)

# Print the prediction
if prediction[0] == 1:
    print("Autism Spectrum Disorder (ASD) is predicted.")
else:
    print("Autism Spectrum Disorder (ASD) is not predicted.")
