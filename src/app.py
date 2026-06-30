import streamlit as st
import pickle
import re
import os
import string
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")



from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ---------------------------------------------------
# Get project paths
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "gru_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "models", "tokenizer.pkl")

# ---------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found:\n{MODEL_PATH}")
    st.stop()

if not os.path.exists(TOKENIZER_PATH):
    st.error(f"Tokenizer file not found:\n{TOKENIZER_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ---------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📱",
    layout="centered"
)

st.title("📱 SMS Spam Detection")
st.write("Enter an SMS message below to check whether it is Spam or Ham.")

message = st.text_area("Enter Message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        cleaned = preprocess_text(message)

        sequence = tokenizer.texts_to_sequences([cleaned])

        padded = pad_sequences(
            sequence,
            maxlen=100,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(padded, verbose=0)

        probability = float(prediction[0][0])

        st.write("Prediction probability:", probability)

        if probability >= 0.5:
            st.error("🚨 Spam Message")
            st.write(f"Confidence: **{probability:.2%}**")
        else:
            st.success("✅ Ham Message")
            st.write(f"Confidence: **{(1-probability):.2%}**")