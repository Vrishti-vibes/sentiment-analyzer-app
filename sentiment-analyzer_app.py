
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Sample dataset
data = {
    "text": [
        "I love this product!",
        "Worst service ever.",
        "Absolutely fantastic experience.",
        "I hate this item.",
        "Not bad, could be better.",
        "Very satisfied with the purchase.",
        "Terrible quality, disappointed.",
        "Excellent support team.",
        "The app is okay.",
        "Poor packaging and slow delivery."
    ],
    "sentiment": [
        "positive", "negative", "positive", "negative", "neutral",
        "positive", "negative", "positive", "neutral", "negative"
    ]
}

# Load and train model
df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']
model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🔍", layout="centered")
st.markdown("<h1 style='text-align: center; color: teal;'>🔍 Sentiment Analyzer App</h1>", unsafe_allow_html=True)
st.markdown("Analyze the sentiment of any text — Positive 😊, Negative 😠, or Neutral 😐")

# Examples
example_texts = [
    "I love this product!",
    "Worst service ever.",
    "It's okay, could be better.",
    "Fantastic customer support!",
    "Very disappointed with the quality."
]

with st.expander("💡 Try Example Texts"):
    for example in example_texts:
        if st.button(example):
            st.session_state.user_input = example

# User input
user_input = st.text_area("✍️ Enter your text here:", value=st.session_state.get("user_input", ""))

if st.button("🔍 Analyze"):
    if user_input.strip():
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]
        confidence = np.max(proba) * 100

        # Emoji feedback
        emoji = "😊" if prediction == "positive" else "😠" if prediction == "negative" else "😐"

        st.markdown(f"### Prediction: **{prediction.capitalize()}** {emoji}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        # Optional: Display probability for all classes
        st.markdown("#### 🔢 Class Probabilities")
        prob_df = pd.DataFrame([proba], columns=model.classes_)
        st.dataframe(prob_df.style.highlight_max(axis=1), use_container_width=True)
    else:
        st.warning("Please enter some text to analyze.")
