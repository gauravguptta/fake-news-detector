import sys
import joblib


# 1. Check for user input

if len(sys.argv) < 2:
    print("⚠️ Usage: python src/predict.py \"Your news headline here\"")
    sys.exit(1)

headline = sys.argv[1]


# 2. Load Model + Vectorizer

model = joblib.load("models/fake_news_model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")


# 3. Prediction

features = vectorizer.transform([headline])
pred = model.predict(features)[0]


# 4. Show Result

print("\n📰 News Headline:", headline)
print("Prediction:", "Real News ✅" if pred == 1 else "Fake News ❌")
