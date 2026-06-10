import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

print("Loading CSV...")

df = pd.read_csv("data/tridis_text_data_full.csv")

print("Columns:")
print(df.columns.tolist())

X = df["text"].fillna("")
y = df["Century"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2)
        )
    ),
    (
        "xgb",
        XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
    )
])

print("Training Century Model...")

model.fit(X, y_encoded)

joblib.dump(
    model,
    "model/century_xgb.pkl"
)

joblib.dump(
    encoder,
    "model/century_encoder.pkl"
)

print("Century Model Saved Successfully")