#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

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
    label = "Real News" if prediction[0] > 0.5 else "Fake News"  # Convert to label
    
    # Display result
    st.write(f"The article is classified as: **{label}**")

    # Optional: Add probability confidence
    confidence = model.predict(input_vector)[0]
    st.write(f"Prediction confidence: {confidence * 100:.2f}%")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script app2.ipynb')

