import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🗳️ Election Prediction App")

st.write("Select details:")

district = st.selectbox("Select District", ["District1", "District2", "District3"])
type_val = st.selectbox("Select Type", ["GEN", "SC", "ST"])

votes = st.number_input("Total Votes")
poll = st.number_input("Poll %")

# 🔥 Add this mapping (you can update later)
party_map = {
    0: "BJP",
    1: "INC",
    2: "Others",
    3: "TMC"
}

if st.button("Predict"):
    prediction = model.predict([[0, 0, votes, poll]])
    
    party_name = party_map.get(prediction[0], "Unknown")
    
    st.success(f"Predicted Winning Party: {party_name}")
