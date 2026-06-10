import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

print("Loading CSV...")

df = pd.read_csv("data/tridis_text_data.csv")

X = df["text"].fillna("")
y = df["Language"]

print("Training Samples:", len(X))

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("rf", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

print("Training Random Forest...")

model.fit(X, y)

joblib.dump(
    model,
    "model/language_rf.pkl"
)

print("language_rf.pkl saved")