# 📰 TrueCheck AI – Fake News Detection System

TrueCheck AI is a Machine Learning based web application that helps detect whether a news article is **Real** or **Fake**.  
The system uses **Natural Language Processing (NLP)** and a trained **PassiveAggressiveClassifier** model to analyze the given news content and provide instant predictions.

This project is designed with a simple and user-friendly interface using **Streamlit**.

---

## 🌐 Live Demo

🔗 https://fake-news-detector-rniugz3wm24xufbmi7kuxj.streamlit.app/

---

## 🚀 Features

✅ Detect Fake and Real News Instantly  
✅ Clean and Interactive Web Interface  
✅ Machine Learning Based Prediction  
✅ Fast and Accurate Results  
✅ Easy to Use  
✅ Deployed Online with Streamlit  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  
- Pickle  

---

## 📂 Project Structure

```bash
fake-news-detector/
│── app.py
│── train.py
│── model.pkl
│── vectorizer.pkl
│── news.csv
│── requirements.txt
│── README.md

Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/YOUR-USERNAME/fake-news-detector.git
cd fake-news-detector
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
streamlit run app.py
🧠 How It Works
User enters news headline or article text
Text is cleaned and preprocessed
TF-IDF Vectorizer converts text into numerical format
Trained ML model analyzes the content
Result is shown as:

✅ Real News
❌ Fake News

📸 Screenshot

(Add your project screenshot here)

![App Screenshot](screenshot.png)
🚀 Future Improvements
Add URL based fake news detection
Improve UI/UX design
Add Deep Learning models
Support Multiple Languages
Show Confidence Score
👨‍💻 Developed By

Sai Tarun Tej