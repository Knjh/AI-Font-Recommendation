import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import yake
from transformers import pipeline

#Theme Toggle
st.set_page_config(page_title="Font Recommender", page_icon="🎨", layout="wide")
st.markdown("""
    <style>
        .css-1d391kg { padding-top: 2rem; }
        .stButton>button { width: 100%; }
        .css-1aumxhk { background-color: #1E1E1E !important; }
        .css-1v3fvcr { background-color: white !important; }
    </style>
""", unsafe_allow_html=True)

# Cache Model Loading to Avoid Repeated Reloads
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-mpnet-base-v2')
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=1)
    nlp = spacy.load("en_core_web_sm")
    return model, sentiment_pipeline, emotion_pipeline, nlp

model, sentiment_pipeline, emotion_pipeline, nlp = load_models()

# Cache Embeddings to Prevent Repeated Loading
@st.cache_data
def load_embeddings():
    with open("vectr.pkl", "rb") as f:
        vectr = pickle.load(f)
    df = pd.read_csv("font_dataset.csv")
    
    df["tag"] = df["Categories"] + df["Feeling Tags"] + df["Appearance Tags"] + \
                df["Calligraphy Tags"] + df["Serif Tags"] + df["Sans Serif Tags"] + \
                df["Technology Tags"] + df["Seasonal Tags"]
    
    df = df[["Font Name", "tag"]]
    return df, vectr

df1, vectr = load_embeddings()

# Analyze User Input Text
def analyze_text(text):
    sentiment_labels = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    
    sentiment_result = sentiment_pipeline(text)[0]
    sentiment = sentiment_labels[sentiment_result['label']]

    emotion = emotion_pipeline(text)[0][0]['label']

    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=5)
    keywords = [kw for kw, _ in kw_extractor.extract_keywords(text)]

    doc = nlp(text)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    return {
        "sentiment": sentiment,
        "emotion": emotion,
        "keywords": keywords,
        "named_entities": named_entities
    }

# Find Similar Fonts Based on Text
def similarity_score_bw_vector_input(vector, text):
    input_embed = model.encode([text])
    similarity_score = cosine_similarity(input_embed, vector)
    simi_score = sorted(list(enumerate(similarity_score[0])), reverse=True, key=lambda x: x[1])
    return simi_score

def recommend(simi_score):
    return [df1.iloc[idx[0]]["Font Name"] for idx in simi_score[:5]]

# Navigation Bar
st.sidebar.title("Kanhaiya Jha")
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to", ["Project Description", "Font Recommender", "Contact"])

theme = st.sidebar.radio("🌙 Theme", ["Light", "Dark"], index=0)
if theme == "Dark":
    st.markdown("""
        <style>
            .stApp { background-color: #121212; color: white; }
        </style>
    """, unsafe_allow_html=True)

# Project Description
if section == "Project Description":
    st.title("📌 Project Description")
    st.markdown("""
    ### Introduction
    This Font Recommender System suggests fonts based on user-input text. It analyzes the sentiment, emotion, and keywords in the text and matches them to fonts using cosine similarity.

    ### Data Collection & Preprocessing
    - Google Font dataset is used which is available on githib 
    - Preprocessed the dataset to find out the important features and metadata against fonts to create a tag
    - Tags are merged into a single column for similarity computation.
    
    ### Models Used
    - `all-mpnet-base-v2` for text embedding.
    - `Cardiff NLP Sentiment Model` for sentiment analysis.
    - `GoEmotions` for emotion detection.
    - `YAKE` for keyword extraction.
    
    ### Recommendation Logic
    - User text is processed for sentiment, emotion, and keywords.
    - The input is converted into an embedding.
    - Cosine similarity is computed against precomputed font embeddings.
    - Top 5 most similar fonts are recommended.
    """)

# Font Recommendation System
if section == "Font Recommender":
    st.title("🎨 Font Recommendation System")
    
    user_input = st.text_area("Enter text to find matching fonts:")
    
    if st.button("Recommend Fonts"):
        if user_input:
            result = analyze_text(user_input)
            tag = f"{result['sentiment']} {result['emotion']} " + " ".join(result['keywords'])

            simi_score = similarity_score_bw_vector_input(vectr, tag)
            recommended_fonts = recommend(simi_score)

            st.subheader("📌 Recommended Fonts:")
            for font in recommended_fonts:
                font_url = f"https://fonts.google.com/specimen/{font.replace(' ', '+')}"
                st.markdown(f"- **{font}** [🔗Link]({font_url})", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to get recommendations.")

# 📞 **Contact Information**
if section == "Contact":
    st.title("📞 Contact")
    st.markdown("""
    **About Me:**
    - 🎓 Data Scientist | NLP Enthusiast
    - 💻 Passionate about Data Scienece, AI and ML-based applications
    - 📧 Contact: knj9304@gmail.com
    """)
