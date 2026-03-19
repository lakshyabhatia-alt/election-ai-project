import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))

st.title("🗳️ Election Prediction App")
menu = st.sidebar.selectbox("Menu", ["Home", "Prediction", "Analysis"])
if menu == "Home":
    st.write("Welcome to Election AI Dashboard")
    st.write("Use this app to analyze and predict election results.")

elif menu == "Analysis":
    st.subheader("📊 Data Analysis")
    
    party_counts = df['Party'].value_counts()
    st.bar_chart(party_counts)

elif menu == "Prediction":
    st.subheader("🤖 Prediction")
    
    # 👉 KEEP YOUR EXISTING prediction code here
state = st.selectbox("Select State", ["Assam", "West Bengal"])
model = pickle.load(open('model.pkl', 'rb'))
le_party = pickle.load(open('party_encoder.pkl', 'rb'))
le_district = pickle.load(open('district_encoder.pkl', 'rb'))
le_type = pickle.load(open('type_encoder.pkl', 'rb'))
if state == "Assam":
    df = pd.read_csv('IndiaVotes_AC__Assam_2021 (1).csv')
else:
    df = pd.read_csv('IndiaVotes_AC__West_Bengal_2021 (1).csv')
st.write("Select details:")

st.subheader("🗺️ Constituency Map")

import numpy as np

# generate random coordinates around state
if state == "Assam":
    lat = 26 + np.random.rand(len(df)) * 2
    lon = 91 + np.random.rand(len(df)) * 2
else:
    lat = 22 + np.random.rand(len(df)) * 2
    lon = 88 + np.random.rand(len(df)) * 2

map_data = pd.DataFrame({'lat': lat, 'lon': lon})

st.map(map_data)

import pandas as pd


st.subheader("📊 Party Distribution")

party_counts = df['Party'].value_counts()

fig, ax = plt.subplots()
party_counts.plot(kind='bar', ax=ax)

st.pyplot(fig)

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
