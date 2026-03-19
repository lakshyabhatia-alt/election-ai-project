import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🗳️ Election Prediction App")

st.write("Select details:")

# Example dropdown values (you can expand later)
district = st.selectbox("Select District", ["District1", "District2", "District3"])
type_val = st.selectbox("Select Type", ["GEN", "SC", "ST"])

votes = st.number_input("Total Votes")
poll = st.number_input("Poll %")

if st.button("Predict"):
    # TEMP: using dummy encoding (we'll improve later)
    prediction = model.predict([[0, 0, votes, poll]])
    st.success(f"Predicted Party Code: {prediction[0]}")
