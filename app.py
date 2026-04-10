import re
import streamlit as st
import pickle
import importlib
import sys
from pathlib import Path
from utils import clean_text, preprocess_nlp

# Compatibility shim for older model pickles that reference numpy._core
try:
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = importlib.import_module('numpy.core')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'

st.set_page_config(page_title="Amazon Feedback AI", page_icon="⭐")

NEGATIVE_PATTERN = re.compile(r"\b(bad|broken|terrible|worst|awful|useless|not working|not satisfied|poor quality|do not buy|hate|disappointed|refund)\b", re.I)

def strong_negative(text: str) -> bool:
    return bool(NEGATIVE_PATTERN.search(text))

@st.cache_resource
def load_assets():
    s = pickle.load(open(MODEL_DIR / 'sentiment_model.pkl', 'rb'))
    r = pickle.load(open(MODEL_DIR / 'rating_model.pkl', 'rb'))
    t = pickle.load(open(MODEL_DIR / 'tfidf_vectorizer.pkl', 'rb'))
    return s, r, t

st.title("🛡️ Amazon Automated Feedback System")
st.markdown("Enter customer feedback below to see the AI's analysis.")

try:
    sent_model, rate_model, tfidf = load_assets()
    
    review_input = st.text_area("Customer Review:", placeholder="Type here...", height=150)
    
    if st.button("Run Analysis"):
        if review_input:
            # Process
            cleaned = clean_text(review_input)
            processed = preprocess_nlp(cleaned)
            vec_input = tfidf.transform([processed])
            
            # Predict
            sentiment = sent_model.predict(vec_input)[0]
            rating = rate_model.predict(vec_input)[0]
            probabilities = sent_model.predict_proba(vec_input)[0]
            labels = list(sent_model.classes_)
            neg_conf = float(probabilities[labels.index('Negative')]) if 'Negative' in labels else 0.0
            pos_conf = float(probabilities[labels.index('Positive')]) if 'Positive' in labels else 0.0

            # Rule-based fallback for very explicit negative language
            if sentiment == 'Positive' and strong_negative(review_input):
                sentiment = 'Negative'
                if neg_conf < 0.5:
                    neg_conf = max(neg_conf, 0.65)
                    pos_conf = 1 - neg_conf

            # Display
            st.divider()
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Sentiment")
                if sentiment == "Positive":
                    st.success(f"{sentiment} ({pos_conf*100:.0f}% confidence)")
                elif sentiment == "Negative":
                    st.error(f"{sentiment} ({neg_conf*100:.0f}% confidence)")
                else:
                    st.warning(sentiment)
                
            with c2:
                st.subheader("Predicted Rating")
                st.write(f"### {rating:.1f} / 5.0")
                
            if sentiment == "Negative":
                st.info("💡 **AI Suggestion:** Flag this for customer support follow-up.")
        else:
            st.warning("Please enter a review to analyze.")
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}. Please run 'python train_models.py' first.")
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure this app is running in the same Python environment where the models were created.")

# Dataset Sentiment Analysis Visualizations Section
st.divider()
st.header("📊 Dataset Sentiment Analysis Visualizations")

image_dir = BASE_DIR / 'data analysis'
if image_dir.exists():
    images = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg'))
    if images:
        for img_path in images:
            st.subheader(f"{img_path.stem.replace('_', ' ').title()}")
            st.image(str(img_path), use_column_width=True)
    else:
        st.info("No images found in the 'output images' folder.")
else:
    st.warning("The 'output images' folder does not exist.")