import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Election AI Dashboard", layout="wide")

st.title("🗳️ Election AI Dashboard")
st.markdown("### Analyze and Predict Election Outcomes")
st.markdown("---")

# ---------------- STATE SELECTION ----------------
state = st.selectbox("Select State", ["Assam", "West Bengal"])

# ---------------- LOAD DATA ----------------
if state == "Assam":
    df = pd.read_csv('IndiaVotes_AC__Assam_2021 (1).csv')
else:
    df = pd.read_csv('IndiaVotes_AC__West_Bengal_2021 (1).csv')

# ---------------- LOAD MODEL & ENCODERS ----------------
model = pickle.load(open('model.pkl', 'rb'))
le_party = pickle.load(open('party_encoder.pkl', 'rb'))
le_district = pickle.load(open('district_encoder.pkl', 'rb'))
le_type = pickle.load(open('type_encoder.pkl', 'rb'))

# ---------------- SIDEBAR MENU ----------------
menu = st.sidebar.selectbox("Menu", ["Home", "Analysis", "Prediction"])

# ---------------- HOME ----------------
if menu == "Home":
    st.write("Welcome to Election AI Dashboard")
    st.write("Use this app to analyze and predict election results.")

# ---------------- ANALYSIS ----------------
elif menu == "Analysis":
    st.subheader("📊 Data Analysis")

    party_counts = df['Party'].value_counts()
    st.bar_chart(party_counts)

    st.subheader("📈 Poll % Distribution")
    st.line_chart(df['Poll%'])

    st.subheader("🏆 Top Constituencies by Votes")
    top = df.sort_values(by='Total Votes', ascending=False).head(10)
    st.bar_chart(top.set_index('AC Name')['Total Votes'])

# ---------------- PREDICTION ----------------
elif menu == "Prediction":
    st.subheader("🤖 Prediction")

    # MAP
    st.subheader("🗺️ Constituency Map")

    if state == "Assam":
        lat = 26 + np.random.rand(len(df)) * 2
        lon = 91 + np.random.rand(len(df)) * 2
    else:
        lat = 22 + np.random.rand(len(df)) * 2
        lon = 88 + np.random.rand(len(df)) * 2

    map_data = pd.DataFrame({'lat': lat, 'lon': lon})
    st.map(map_data)

    # GRAPH
    st.subheader("📊 Party Distribution")
    fig, ax = plt.subplots()
    df['Party'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # INPUTS
    district = st.selectbox("Select District", list(le_district.classes_))
    type_val = st.selectbox("Select Type", list(le_type.classes_))

    col1, col2 = st.columns(2)

    with col1:
        votes = st.number_input("Total Votes", min_value=0)

    with col2:
        poll = st.number_input("Poll %", min_value=0.0, max_value=100.0)

    # PREDICTION
    if st.button("Predict"):
        district_encoded = le_district.transform([district])[0]
        type_encoded = le_type.transform([type_val])[0]

        prediction = model.predict([[district_encoded, type_encoded, votes, poll]])
        party_name = le_party.inverse_transform(prediction)

        st.success(f"Predicted Winning Party: {party_name[0]}")
