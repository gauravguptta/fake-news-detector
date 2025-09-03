import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model_and_vectorizer, load_dataset, evaluate_model


# Setup

st.set_page_config(page_title="Fake News Detector", layout="wide")

# Load model & dataset
model, vectorizer = load_model_and_vectorizer("models/fake_news_model.joblib", "models/vectorizer.joblib")
X_test, y_test, data = load_dataset("data/merged_dataset.csv", vectorizer)


# Custom CSS

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {background-color: #f8f9fa; font-family: 'Segoe UI', sans-serif;}
    h1 {color: #1a1a1a; font-size: 28px;}
    h2 {color: #0d6efd; margin-top: 1.5rem;}
    .stButton button {
        background-color: #0d6efd; color: white; border-radius: 6px;
        padding: 0.5rem 1rem; border: none; font-weight: 500;
    }
    .stButton button:hover {background-color: #0b5ed7;}
    </style>
""", unsafe_allow_html=True)


# UI

st.title("Fake News Detector")
st.write("Check whether news is Real or Fake with live API integration, custom input, and model evaluation.")

st.sidebar.header("Model Info")
st.sidebar.write("- Logistic Regression")
st.sidebar.write("- Dataset size: ~20,000")
st.sidebar.write("- Accuracy: ~94%")

# 1. Single Prediction
st.header("Single Headline Prediction")
headline = st.text_input("Enter a news headline")
if st.button("Predict"):
    if headline.strip():
        vec = vectorizer.transform([headline])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][pred] * 100
        if pred == 1:
            st.success(f"Real News ({prob:.2f}% confidence)")
            st.progress(int(prob))
        else:
            st.error(f"Fake News ({prob:.2f}% confidence)")
            st.progress(int(prob))
    else:
        st.warning("Please enter a headline.")

# 2. Batch Prediction
st.header("Batch Testing")
batch_input = st.text_area("Paste multiple headlines:")
if st.button("Run Batch Prediction"):
    if batch_input.strip():
        headlines = [line.strip() for line in batch_input.split("\n") if line.strip()]
        vecs = vectorizer.transform(headlines)
        preds = model.predict(vecs)
        probs = model.predict_proba(vecs).max(axis=1) * 100
        results = pd.DataFrame({
            "Headline": headlines,
            "Prediction": ["Real" if p == 1 else "Fake" for p in preds],
            "Confidence": [f"{p:.2f}%" for p in probs]
        })
        st.dataframe(results, use_container_width=True)

# 3. Live News
st.header("Latest News (API)")
if st.button("Fetch Latest News"):
    API_KEY = "580cd64a949f415a8e0c1990cc451ca1"
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        headlines = [a["title"] for a in articles if a["title"]][:10]
        if headlines:
            vecs = vectorizer.transform(headlines)
            preds = model.predict(vecs)
            probs = model.predict_proba(vecs).max(axis=1) * 100
            results = pd.DataFrame({
                "Headline": headlines,
                "Prediction": ["Real" if p == 1 else "Fake" for p in preds],
                "Confidence": [f"{p:.2f}%" for p in probs]
            })
            st.dataframe(results, use_container_width=True)

# 4. Model Evaluation
st.header("Model Evaluation")
if st.button("Show Confusion Matrix & Report"):
    cm, report = evaluate_model(model, X_test, y_test)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)
