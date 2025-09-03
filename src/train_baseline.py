import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset

data_path = "data/train.csv"
df = pd.read_csv(data_path)

# We’ll use only the headline (title) for now
X = df["title"]
y = df["label"]


# 2. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3. Text Vectorization

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 4. Model Training

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)


# 5. Evaluation

y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print("\n🎯 Training complete")
print(f"Model Accuracy: {acc:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 6. Visualization

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# 7. Save Model + Vectorizer

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/fake_news_model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")
print("\n✅ Model and vectorizer stored in 'models/' folder")


# 8. Quick Test

sample = ["Breaking news: Government launches new education policy"]
sample_vec = vectorizer.transform(sample)
pred_label = clf.predict(sample_vec)[0]
print("\n🔎 Example Prediction:", "Real" if pred_label == 1 else "Fake")
