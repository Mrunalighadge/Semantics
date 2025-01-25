
import streamlit as st
import joblib
#Load the saved model and vectorizer\n",
model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('random_forest_model.pkl')
#Title and description
st.title("Fake News Classifier")    
# Input box for user\n",
user_input = st.text_area("Enter a news article here:", height=200)
 # Predict button
if st.button("Predict"):
# Preprocess and transform input\n",
    input_vector = vectorizer.transform([user_input])  # Transform input using TF-IDF\n",
    prediction = model.predict(input_vector)          # Predict using the trained model\n",
    if prediction[0] == 1:
        st.write("this is a real news")
    else:
        st.write("this is a fake news")
else:
    st.write("enter some text first")