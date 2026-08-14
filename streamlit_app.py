"""
IMDB Sentiment Analysis - Streamlit Application

Responsibilities:
1. Load sentiment prediction function
2. Take user review input
3. Display prediction result
"""


import streamlit as st

from src.predict import predict_sentiment


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬"
)


# ==================================================
# TITLE
# ==================================================

st.title(
    "🎬 IMDB Movie Review Sentiment Analysis"
)

st.write(
    "Enter a movie review and predict whether it is positive or negative."
)


# ==================================================
# USER INPUT
# ==================================================

review = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Write your movie review here..."
)


# ==================================================
# PREDICTION
# ==================================================

if st.button("Predict Sentiment"):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        sentiment, probability = predict_sentiment(
            review
        )


        st.subheader(
            "Prediction Result"
        )


        if sentiment == "Positive":

            st.success(
                "😊 Positive Review"
            )

        else:

            st.error(
                "😞 Negative Review"
            )


        st.write(
            f"Confidence: {probability:.2%}"
        )