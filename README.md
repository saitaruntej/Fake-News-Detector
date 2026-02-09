# 🕵️‍♀️ TrueCheck AI - Fake News Detector

**TrueCheck AI** is a machine learning-powered application designed to detect fake news articles based on their content. It uses a **PassiveAggressiveClassifier** trained on a dataset of real and fake news to classify articles with high accuracy.

The project features a beautiful, user-friendly web interface built with **Streamlit** and a command-line interface for quick predictions.

## ✨ Features

- **Real-time Fake News Detection**: Instantly analyze news articles and get a verification result (Real vs. Fake).
- **Interactive Web App**: A modern, responsive UI with a dark-themed design.
- **Search & Auto-fill**: Select existing headlines from the dataset to test the model.
- **High Accuracy**: Utilizes TF-IDF vectorization and a PassiveAggressiveClassifier for robust text classification.
- **Command Line Tools**: Scripts for training the model and running predictions from the terminal.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saitaruntej/Fake-News-Detector.git
    cd Fake-News-Detector
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r streamlit.txt
    ```

## 🛠️ Usage

### 1. Run the Web Application (Recommended)
Launch the interactive web interface:
```bash
streamlit run app.py
```
Or:
```bash
python -m streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

### 2. Run CLI Prediction
Test the model directly from your terminal:
```bash
python predict.py
```

### 3. Train the Model (Optional)
If you want to retrain the model with new data (place your `news.csv` in the root directory):
```bash
python train.py
```

## 📂 Project Structure

- `app.py`: Main Streamlit web application.
- `train.py`: Script to train the machine learning model.
- `predict.py`: CLI script for testing predictions.
- `model.pkl`: Pre-trained model file.
- `vectorizer.pkl`: Pre-trained TF-IDF vectorizer.
- `news.csv`: Dataset used for training and testing.
- `streamlit.txt`: List of Python dependencies.

## 👨‍💻 Credits

Developed by **Sai Teja**.
