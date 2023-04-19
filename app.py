import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load the model and tokenizer
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Define the function that runs the model
def predict_sentiment(text):
    encoded_text = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'Negative' : scores[0],
        'Neutral' : scores[1],
        'Positive' : scores[2]
    }
    return scores_dict

# Define the function that classifies the sentiment based on the scores
def classify_sentiment(scores_dict):
    if scores_dict['Negative'] > scores_dict['Neutral'] and scores_dict['Negative'] > scores_dict['Positive']:
        return "You may have suicidal tendencies."
    else:
        return "You do not appear to have suicidal tendencies."

# Create the Streamlit app
st.set_page_config(page_title="Suicidal Tendencies Detection")
st.title("Suicidal Tendencies Detection")
st.markdown("Enter the text you want to analyze below:")
user_input = st.text_input("Enter text")
submit_button = st.button("Submit")

# Run the app and display the results
if submit_button:
    result = predict_sentiment(user_input)
    classification = classify_sentiment(result)
    st.write("### Results:")
    st.write(classification)
