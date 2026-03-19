import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Election Prediction App")

st.write("Enter details below:")

district = st.number_input("District (encoded)")
type_val = st.number_input("Type (encoded)")
votes = st.number_input("Total Votes")
poll = st.number_input("Poll %")

if st.button("Predict"):
    prediction = model.predict([[district, type_val, votes, poll]])
    st.success(f"Predicted Party Code: {prediction[0]}")
