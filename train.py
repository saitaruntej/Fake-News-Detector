import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

print("Loading and upgrading dataset...")
try:
    df = pd.read_csv('news.csv')
    
    # IMPROVEMENT: Combine 'title' and 'text' so the model reads the Headline too!
    # This helps catch clickbait headlines which are often fake.
    df['combined_text'] = df['title'] + " " + df['text']
    
    x = df['combined_text']
    y = df['label']
    
except Exception as e:
    print(f"Error: {e}")
    exit()

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Vectorize (with slightly better settings)
# min_df=5 means "ignore words that appear in fewer than 5 documents" (removes typos/noise)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)

tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# Train
print("Training upgraded model...")
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Evaluate
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'New Model Accuracy: {round(score*100,2)}%')

# Save
with open('model.pkl', 'wb') as model_file:
    pickle.dump(pac, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Success! Upgraded model saved.")