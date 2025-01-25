#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pickle

# Load the saved model and vectorizer using pickle
with open('random_forest_model.pkl'(1), 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Title of the app
st.title("Fake News Classifier")
st.write("This app predicts whether a given news article is **Real** or **Fake**.")

# Input text box
user_input = st.text_area("Enter a news article:", height=200)

# Prediction button
if st.button("Predict"):
    # Preprocess and predict
    input_vector = vectorizer.transform([user_input])  # Transform input using TF-IDF
    prediction = model.predict(input_vector)  # Get prediction
    label = "Real News" if prediction[0] == 1 else "Fake News"  # Convert to label
    
    # Display result
    st.write(f"The article is classified as: **{label}**")

    # Optional: Add probability confidence (if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_vector)  # Get probabilities
        confidence = prob[0][1] if prediction[0] == 1 else prob[0][0]  # Confidence for the predicted class
        st.write(f"Prediction confidence: {confidence * 100:.2f}%")
    else:
        st.write("Model does not support confidence scores.")
