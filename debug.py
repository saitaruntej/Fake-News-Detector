import pandas as pd
import pickle

print("--- DIAGNOSTIC REPORT ---")

# 1. Check the CSV Data
try:
    df = pd.read_csv('news.csv')
    print(f"1. Total Rows in CSV: {len(df)}")
    print(f"2. Columns Found: {list(df.columns)}")

    # Check if the 'label' column exists
    if 'label' in df.columns:
        print("\n3. Label Counts (What the model learned from):")
        print(df['label'].value_counts())
    else:
        print("\nERROR: 'label' column missing!")

except Exception as e:
    print(f"Error reading CSV: {e}")

# 2. Check the Model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"\n4. Model Classes: {model.classes_}")
except:
    print("Error loading model.")

print("-------------------------")