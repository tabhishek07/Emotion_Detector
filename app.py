import streamlit as st
from emotion_detector import predict_emotion


# Page config
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧐",
    layout="centered"
)


st.title("🧐 Emotion Detector")
st.write("""
Enter any text below and click **Detect Emotion**.  
This app uses a TF-IDF + Logistic Regression model
trained on tweets to predict one of six emotions,
with heuristic overrides for greetings & profanity.
""")


user_input = st.text_area(
    "Your text here:",
    placeholder="Type or paste a sentence...",
    height=150
)


if st.button("Detect Emotion"):
    if not user_input.strip():
        st.error("Please enter some text first.")
    else:
        # Get prediction + probabilities
        emotion, probs = predict_emotion(user_input)
        st.success(f"**Detected emotion:** {emotion.upper()}")
        st.write("**Confidence by class:**")
        for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
            st.write(f"- {emo.title():<10}: {p:.2%}")
