import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained('model_spam')
    tokenizer = BertTokenizer.from_pretrained('model_spam')
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app layout
st.title("Email Spam Classifier")
st.write("Enter the email content below, and the app will predict if it's spam or not.")

# Input field for email content
email_content = st.text_area("Email Content")

# Classification function
def classify_email(text):
    # Tokenize and prepare the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the prediction
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Spam" if prediction == 1 else "Not Spam"

# Button to classify email
if st.button("Classify"):
    if email_content:
        result = classify_email(email_content)
        st.write(f"The email is classified as: **{result}**")
    else:
        st.write("Please enter some text in the email content field.")


# subject: Buy this ticket to win lottery

# Buy this ticket to win 1 million dollars lottery