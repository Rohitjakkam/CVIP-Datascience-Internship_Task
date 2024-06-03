import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load vectorizer and model from pickle files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit interface
st.title("Golden Task Spam Classifier")

# Text input
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)
    # Vectorize the text
    vector_input = tfidf.transform([transformed_sms])
    # Predict using the model
    result = model.predict(vector_input)[0]
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
