{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a1189ff3-7627-4d92-af4e-ff56ee6004b5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-01-25 14:25:32.402 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\dell\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "\n",
    "# Load the saved model and vectorizer\n",
    "model = joblib.load('random_forest_model.pkl')\n",
    "vectorizer = joblib.load('random_forest_model.pkl')\n",
    "\n",
    "# Title and description\n",
    "st.title(\"Fake News Classifier\")\n",
    "st.write(\"\"\"\n",
    "This web application predicts whether a given news article is **Fake** or **Real**. \n",
    "Enter a news article in the text box below and click **Predict**.\n",
    "\"\"\")\n",
    "\n",
    "# Input box for user\n",
    "user_input = st.text_area(\"Enter a news article here:\", height=200)\n",
    "\n",
    "# Predict button\n",
    "if st.button(\"Predict\"):\n",
    "    # Preprocess and transform input\n",
    "    input_vector = vectorizer.transform([user_input])  # Transform input using TF-IDF\n",
    "    prediction = model.predict(input_vector)          # Predict using the trained model\n",
    "    label = \"Real News\" if prediction[0] == 1 else \"Fake News\"  # Convert to label\n",
    "    \n",
    "    # Display result\n",
    "    st.write(f\"The article is classified as: **{label}**\")\n",
    "\n",
    "    # Optional: Add confidence/probability\n",
    "    probability = model.predict_proba(input_vector)[0]\n",
    "    st.write(f\"Confidence: {probability.max() * 100:.2f}%\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
