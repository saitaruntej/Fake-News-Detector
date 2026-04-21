import pickle

def detect_news():
    print("--- Fake News Detector ---")
    print("Loading model...")

    # Load the saved model and vectorizer
    try:
        with open('model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)
    except FileNotFoundError:
        print("Error: Model files not found. Run 'train.py' first!")
        return

    # Interaction Loop
    while True:
        print("\n-------------------------------------------------")
        news_text = input("Enter news text to check (or type 'quit' to exit):\n")

        if news_text.lower() == 'quit':
            break

        if not news_text.strip():
            continue

        # Transform input and Predict
        vec_text = loaded_vectorizer.transform([news_text])
        prediction = loaded_model.predict(vec_text)

        print(f"\nResult: {prediction[0]}")

if __name__ == "__main__":
    detect_news()