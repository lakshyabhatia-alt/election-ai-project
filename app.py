import streamlit as st
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
le_party = pickle.load(open('party_encoder.pkl', 'rb'))
le_district = pickle.load(open('district_encoder.pkl', 'rb'))
le_type = pickle.load(open('type_encoder.pkl', 'rb'))

st.title("🗳️ Election Prediction App")

st.write("Select details:")

# REAL dropdown values
district = st.selectbox("Select District", list(le_district.classes_))
type_val = st.selectbox("Select Type", list(le_type.classes_))

votes = st.number_input("Total Votes", min_value=0)
poll = st.number_input("Poll %", min_value=0.0, max_value=100.0)

if st.button("Predict"):
    # Convert input → encoded
    district_encoded = le_district.transform([district])[0]
    type_encoded = le_type.transform([type_val])[0]

    prediction = model.predict([[district_encoded, type_encoded, votes, poll]])
    
    party_name = le_party.inverse_transform(prediction)
    
    st.success(f"Predicted Winning Party: {party_name[0]}")
