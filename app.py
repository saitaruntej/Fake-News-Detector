import streamlit as st
import pandas as pd
import pickle

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="TrueCheck AI",
    page_icon="🕵️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS FOR STYLING
st.markdown("""
    <style>
    /* Main background image */
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                          url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #FAFAFA;
    }
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 20, 30, 0.6); 
        backdrop-filter: blur(10px);
    }
    
    /* Input Text Box Styling */
    .stTextArea>div>div>textarea {
        background-color: rgba(0, 0, 0, 0.5);
        color: #FFFFFF;
        border: 1px solid #4F4F4F;
        border-radius: 10px;
    }
    
    /* Button Styling */
    .stButton>button {
        color: white;
        background-color: #FF4B4B;
        border: none;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6C6C;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. LOAD MODEL & DATA
@st.cache_resource
def load_resources():
    # Load Model
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
    except FileNotFoundError:
        return None, None, None

    # Load Dataset for Recommendations
    try:
        df = pd.read_csv('news.csv')
        # Ensure we have titles and text
        if 'title' in df.columns and 'text' in df.columns:
            # Create a dictionary for quick lookup {title: text}
            news_dict = pd.Series(df.text.values, index=df.title).to_dict()
        else:
            news_dict = {}
    except Exception:
        news_dict = {}
        
    return model, vectorizer, news_dict

model, vectorizer, news_data = load_resources()

# Initialize session state for text area
if 'news_content' not in st.session_state:
    st.session_state['news_content'] = ""

# 4. SIDEBAR OPTIONS
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910768.png", width=80)
    st.title("TrueCheck AI")
    st.write("### ⚙️ Settings")
    st.info("Select a news headline from the dropdown to auto-fill the text, or type your own.")
    st.write("---")
    st.caption(f"Developed by: **Sai Teja**")

# 5. MAIN CONTENT
st.title("🕵️‍♀️ Fake News Detector")
st.markdown("### Verify the authenticity of news articles in seconds.")

# --- SEARCH / RECOMMENDATION BAR ---
# This is the new feature: Searchable Dropdown
if news_data:
    st.write("#### 🔍 Search & Select News (Optional)")
    # Add a default "empty" option
    options = ["Type or Select a Headline..."] + list(news_data.keys())
    
    selected_headline = st.selectbox(
        "Start typing to search existing news:",
        options=options,
        label_visibility="collapsed"
    )

    # If user selects a real headline, update the text area
    if selected_headline != "Type or Select a Headline...":
        st.session_state['news_content'] = news_data[selected_headline]

# --- MAIN LAYOUT ---
col1, col2 = st.columns([3, 1])

with col1:
    # Text Area gets value from session_state
    news_text = st.text_area(
        "News Content:", 
        value=st.session_state['news_content'],
        height=300, 
        placeholder="Paste the text you want to analyze or select from the search bar above..."
    )

with col2:
    st.write("### Analysis")
    analyze_button = st.button("🔍 Analyze News")
    
    if st.button("🗑️ Clear Text"):
        st.session_state['news_content'] = "" # Clear session state
        st.rerun()

# 6. PREDICTION LOGIC
if analyze_button:
    if model is None:
        st.error("⚠️ Model not found! Please run 'train.py' first.")
    elif not news_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text patterns..."):
            vec_text = vectorizer.transform([news_text])
            prediction = model.predict(vec_text)[0]
            
            st.write("---")
            if prediction == "FAKE":
                st.error(f"🚨 **RESULT: FAKE NEWS DETECTED**")
                st.write("The AI found patterns consistent with misinformation.")
            else:
                st.success(f"✅ **RESULT: REAL NEWS**")
                st.write("This article appears to be legitimate.")