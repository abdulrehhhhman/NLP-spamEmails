# app.py

import streamlit as st
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("📩 Spam Message Classifier")
st.write("Enter an SMS message below and find out if it's spam or not.")

# Input
user_input = st.text_area("Enter your message here")

# Predict
if st.button("Check"):
    result = model.predict([user_input])[0]
    if result == 1:
        st.error("🚨 This message is **SPAM**.")
    else:
        st.success("✅ This message is **HAM** (not spam).")
