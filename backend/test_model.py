import joblib

language_model = joblib.load(
    "model/language_rf.pkl"
)

script_model = joblib.load(
    "model/script_xgb.pkl"
)

script_encoder = joblib.load(
    "model/script_encoder.pkl"
)

text = "dictus Hugo in extrema volun"

language = language_model.predict([text])[0]

script_id = script_model.predict([text])[0]

script = script_encoder.inverse_transform(
    [script_id]
)[0]

print("Language:", language)
print("Script:", script)