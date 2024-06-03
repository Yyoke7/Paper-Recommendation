from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import faiss

@st.cache_resource
def compute_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['title'] + " " + df['abstract'])
    return vectorizer, tfidf_matrix

@st.cache_resource
def preprocess_title(title):
    return ' '.join(title.strip().lower().split())

@st.cache_resource
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index