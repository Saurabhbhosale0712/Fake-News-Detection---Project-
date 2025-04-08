import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("🧠 Fake News Detection App")
st.markdown("Enter a news headline or article text and check if it's **Real** or **Fake**!")

user_input = st.text_area("📰 Enter news text here:", height=200)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some news text to detect.")
    else:
        cleaned_text = preprocess(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        
        if prediction == 1:
            st.success("✅ This looks like **Real News**.")
        else:
            st.error("🚨 This appears to be **Fake News**.")
