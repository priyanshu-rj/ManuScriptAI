import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

df = pd.read_csv("data/tridis_text_data_full.csv")

X = df["text"].fillna("")
y = df["Century"]

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("xgb", XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    ))
])

model.fit(X, y_encoded)

joblib.dump(
    model,
    "model/century_xgb.pkl"
)

joblib.dump(
    encoder,
    "model/century_encoder.pkl"
)

print("century_xgb.pkl saved")