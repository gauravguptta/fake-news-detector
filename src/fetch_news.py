import requests
import joblib


# Load trained model & vectorizer

model = joblib.load("models/fake_news_model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")


# NewsAPI Configuration

API_KEY = "580cd64a949f415a8e0c1990cc451ca1"
URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"


# Fetch latest news

response = requests.get(URL)

if response.status_code == 200:
    data = response.json()
    articles = data.get("articles", [])

    if not articles:
        print("⚠️ No news articles found.")
    else:
        print(f"📰 Found {len(articles)} headlines. Running predictions...\n")
        for i, article in enumerate(articles[:10], start=1):  # Top 10 news
            headline = article["title"]
            vec = vectorizer.transform([headline])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0][pred] * 100

            label = "Real" if pred == 1 else "Fake"
            print(f"{i}. {headline}")
            print(f"   → Prediction: {label} ({prob:.2f}% confidence)\n")
else:
    print("❌ Error fetching news:", response.status_code)
