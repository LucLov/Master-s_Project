import joblib

MODEL_PATH = "classification/model.pkl"

def main():
    pack = joblib.load(MODEL_PATH)
    vectorizer = pack["vectorizer"]
    mlb = pack["binarizer"]
    model = pack["model"]

    while True:
        text = input("\nEnter text to classify: ").strip()
        if not text:
            break

        X = vectorizer.transform([text])
        pred = model.predict(X)
        labels = mlb.inverse_transform(pred)[0]

        print("Predicted labels:", labels)


if __name__ == "__main__":
    main()
