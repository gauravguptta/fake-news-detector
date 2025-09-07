import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_model_and_vectorizer, load_dataset, evaluate_model


# Page Setup

st.set_page_config(page_title="Fake News Detector", layout="wide")

# Custom CSS for background and style
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background: linear-gradient(135deg, #e0f2fe, #3b82f6);
        font-family: 'Segoe UI', sans-serif;
        color: #1e293b;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton button {
        background-color: #0d6efd;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #0b5ed7;
    }
    </style>
""", unsafe_allow_html=True)


# Load model and dataset

model, vectorizer = load_model_and_vectorizer(
    "models/fake_news_model.joblib",
    "models/vectorizer.joblib"
)
X_test, y_test, data = load_dataset("data/merged_dataset.csv", vectorizer)


# UI Sections

st.title("Fake News Detector")
st.write("Test whether a news headline is **Real** or **Fake** using a trained ML model.")

# Sidebar Info
st.sidebar.header("Model Info")
st.sidebar.write("- Algorithm: Logistic Regression")
st.sidebar.write("- Dataset size: ~20,000 records")
st.sidebar.write("- Accuracy: ~94%")


# 1. Single Prediction

st.header("Single Headline Prediction")
headline = st.text_input("Enter a news headline:")
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


# 2. Batch Testing

st.header("Batch Testing")
batch_input = st.text_area("Paste multiple headlines (one per line):")
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
    else:
        st.warning("Enter at least one headline.")


# 3. Live News (API)

st.header("Latest News (via NewsAPI)")
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
        else:
            st.warning("No news found.")
    else:
        st.error(f"API Error: {response.status_code}")


# 4. Model Evaluation

st.header("Model Evaluation")
if st.button("Show Confusion Matrix & Report"):
    from sklearn.metrics import confusion_matrix, classification_report

    # Predictions
    y_pred = model.predict(X_test)

    # Confusion Matrix (compact)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"],
                cbar=False, annot_kws={"size": 10}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12)
    st.pyplot(fig)

    # Classification Report (compact)
    report = classification_report(
        y_test, y_pred,
        target_names=["Fake", "Real"],
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose().iloc[:-1, :-1]
    st.subheader("Classification Report")
    st.dataframe(
        report_df.style.background_gradient(cmap="Blues"),
        use_container_width=True,
        height=200
    )

