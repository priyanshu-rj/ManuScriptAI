# this code is for checking model accuracy
import joblib

model = joblib.load("./model/language_rf.pkl")
# model is train on historical manuscripts data not english so they didnot predict eng well
tests = [
    "Hello my name is Priyanshu",
    "This is an English sentence",
    "Bonjour tout le monde",
    "Salve amice"
]

for t in tests:
    print(t)
    print(model.predict([t])[0])
    print()