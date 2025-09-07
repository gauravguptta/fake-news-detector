*Fake News Detection (ML + NLP Project)*

This repository contains an end-to-end pipeline for detecting fake news using Natural Language Processing and Machine Learning.
The implementation covers preprocessing, model training, evaluation, and deployment steps.

*Features*

-Train baseline models using TF-IDF features + Logistic Regression
-Evaluate model performance with accuracy, confusion matrix, and classification report
-Save trained model and vectorizer for later use
-Extendable to more advanced models (SVM, Naive Bayes, XGBoost, etc.)
-Ready for integration with FastAPI / Streamlit for deployment.

*Setup Instructions*

*1. Clone the repository*
git clone <https://github.com/gauravguptta/fake-news-detector>
cd fake-news-detector

*2. Create & activate a virtual environment*
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or CMD
.\.venv\Scripts\activate.bat
# or Git Bash
source .venv/Scripts/activate

*3. Install dependencies*
pip install --upgrade pip
pip install -r requirements.txt

*Project Structure*
fake-news-detector/
│── data/                 # datasets
│   └── sample/           # small sample CSV for quick testing
│── models/               # trained models will be saved here
│── reports/              # metrics and plots
│── src/                  # source code
│   └── train_baseline.py
│── requirements.txt
└── README.md

*Quick Test (with sample dataset)*
python src/train_baseline.py --data_path data/sample/sample_train.csv --include_title

This will:
Train a baseline Logistic Regression model
Save the trained model in models/
Generate evaluation metrics in reports/

*Using the Kaggle Dataset*
To use the full Kaggle Fake News dataset:
Download the dataset (train.csv).
Place it under data/train.csv.
Run the training script:
     python src/train_baseline.py --data_path data/train.csv --include_title


*Expected Columns: id, title, author, text, label*

label: 1 = FAKE, 0 = REAL

*Tech Stack*
- Python 3.10
- Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- Joblib for model persistence
- Streamlit for UI
- FastAPI for API deployment (planned)

*Next Steps*
- Enhance preprocessing (remove URLs, emojis, HTML tags)
- Try advanced ML models (SVM, Naive Bayes, XGBoost)
- Hyperparameter tuning with GridSearchCV
- Integrate real-time news API for live predictions
- Deploy on cloud (Render / AWS / Heroku)

*Acknowledgments*
- Dataset: Kaggle Fake News Detection Challenge
