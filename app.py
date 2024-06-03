import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import requests
from bs4 import BeautifulSoup
import browserhistory as hb
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import faiss
from data_loader import load_data, load_embeddings
from model_loader import get_knn_model, load_model
from preprocessing import compute_tfidf, preprocess_title, create_faiss_index
from recommendation import get_recommendations_knn, get_recommendations_tfidf, get_recommendations_faiss
import base64

FILE = './data/arxiv-metadata-oai-snapshot.json'
FILE_EMBEDDINGS = './data/embeddings.pkl'

st.set_page_config(page_title="论文推荐系统", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 8px 20px;
        margin: 5px;
        border: none;
        border-radius: 12px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar-content {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        color: #4CAF50;
    }
    .sidebar-content img {
        border-radius: 50%;
        margin-right: 10px;
    }
    .title-style {
        font-family: 'Arial', sans-serif;
        color: #4CAF50;
        margin-top: 20px;
    }
    .abstract-style {
        font-family: 'Arial', sans-serif;
        color: #333;
        margin-bottom: 20px;
    }
    .recommended-paper {
        background-color: #fff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .recommended-paper:hover {
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User Icon and Username
st.sidebar.markdown(
    """
    <div class="sidebar-content">
        <img src="https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png" width="40" height="40">
        <span>麻瓜看不到此用户@Group26</span>
    </div>
    """,
    unsafe_allow_html=True
)

def sidebar_bg(side_bg):
    side_bg_ext = 'jpeg'
    with open(side_bg, "rb") as image_file:
        side_bg_base64 = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/{side_bg_ext};base64,{side_bg_base64}) no-repeat bottom;
            background-size: cover;
            padding-bottom: 1;
            margin-bottom: 0;
        }}
        [data-testid="stSidebar"] > div:first-child > div {{
            background-color: rgba(255, 255, 255, 0.8); /* Adjust the background color and opacity */
            padding-bottom: 200px; /* Adjust this value if needed */
        }}
        </style>
        """,
        unsafe_allow_html=True)
sidebar_bg('images/20200530162601_zkYVl.jpeg')

@st.cache_resource
def load_all_data():
    df = load_data(FILE)
    embeddings = load_embeddings(FILE_EMBEDDINGS)
    nn_model = get_knn_model(embeddings)
    sentence_encoder_layer = load_model()
    vectorizer, tfidf_matrix = compute_tfidf(df)
    index = create_faiss_index(embeddings)
    return df, embeddings, nn_model, sentence_encoder_layer, vectorizer, tfidf_matrix, index

df, embeddings, nn_model, sentence_encoder_layer, vectorizer, tfidf_matrix, index = load_all_data()

st.title("📚 论文推荐系统")
st.sidebar.title("设置")
algorithm = st.sidebar.selectbox("选择推荐算法", ["KNN", "TF-IDF", "FAISS"])

option = st.sidebar.selectbox("选择输入方式", ["自行输入论文信息", "自动获取浏览记录"])

def close_chrome():
    for proc in psutil.process_iter():
        if 'chrome' in proc.name().lower():
            proc.kill()

if option == "自行输入论文信息":
    st.header("输入新的论文信息")

    input_titles = []
    input_abstracts = []
    num_papers = st.number_input("输入论文数量：", min_value=1, value=1)

    for i in range(num_papers):
        st.subheader(f"论文 {i + 1}")
        input_title = st.text_input(f"论文 {i + 1} 标题：", key=f"title_{i}")
        input_abstract = st.text_area(f"论文 {i + 1} 摘要：", key=f"abstract_{i}")
        input_titles.append(input_title)
        input_abstracts.append(input_abstract)

    if st.button("获取推荐"):
        if all(input_titles) or all(input_abstracts):
            viewed_embeddings = []
            viewed_titles = []
            for title, abstract in zip(input_titles, input_abstracts):
                processed_title = preprocess_title(title)
                input_text = processed_title + " " + abstract
                if algorithm == "KNN":
                    input_embedding = sentence_encoder_layer([input_text]).numpy()[0]
                    viewed_embeddings.append(input_embedding)
                    viewed_titles.append(processed_title)
                elif algorithm == "TF-IDF":
                    input_tfidf = vectorizer.transform([input_text])
                elif algorithm == "FAISS":
                    input_embedding = sentence_encoder_layer([input_text]).numpy()[0]
                    viewed_embeddings.append(input_embedding)
                    viewed_titles.append(processed_title)
            
            if algorithm == "KNN":
                avg_embedding = np.mean(viewed_embeddings, axis=0)
                recommended_papers = get_recommendations_knn(avg_embedding, nn_model, df, embeddings, viewed_titles)
            elif algorithm == "TF-IDF":
                avg_text = " ".join([title + " " + abstract for title, abstract in zip(input_titles, input_abstracts)])
                recommended_papers = get_recommendations_tfidf(avg_text, vectorizer, tfidf_matrix, df, viewed_titles)
            elif algorithm == "FAISS":
                avg_embedding = np.mean(viewed_embeddings, axis=0)
                recommended_papers = get_recommendations_faiss(avg_embedding, index, df, viewed_titles)

            if recommended_papers:
                st.markdown(f"""
                    <div class="recommended-paper">
                            <h3 class="title-style" style="color: brown;">推荐的论文列表：</h3>
                        </div>
                    """, unsafe_allow_html=True)
                for paper in recommended_papers:
                    st.markdown(f"""
                    <div class="recommended-paper">
                            <h3 class="title-style" style="color: brown;">{paper['title']} ({paper['year']})</h3>
                            <p class="abstract-style">摘要：{paper['abstract']}</p>
                            <p class="abstract-style">相似度：{paper['similarity_score']:.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)

                fig, ax = plt.subplots()
                titles = [paper['title'] for paper in recommended_papers]
                similarities = [paper['similarity_score'] for paper in recommended_papers]
                ax.barh(titles, similarities, color='skyblue')
                ax.set_xlabel('Similarity')
                ax.set_xlim(0, 1)
                st.pyplot(fig)
            else:
                st.write("没有推荐的论文")
        else:
            st.write("请填写所有字段")
else:
    if st.button("自动获取浏览记录并推荐"):
        close_chrome()
        browser_history = hb.get_browserhistory()
        df_history = pd.DataFrame(browser_history['chrome'])
        st.markdown(f"""
                    <div class="recommended-paper">
                            <h3 class="title-style">浏览记录：</h3>
                        </div>
                    """, unsafe_allow_html=True)

        if not df_history.empty:
            viewed_embeddings = []
            input_titles = []
            input_abstracts = []
            count = 0
            for i in range(len(df_history)):
                url = df_history.iloc[i][0]
                if url.startswith("https://arxiv.org/abs/"):
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title_tag = soup.find('h1', class_='title mathjax')
                    abstract_tag = soup.find('blockquote', class_='abstract mathjax')
                    title = title_tag.text.strip().replace('Title:', '')
                    abstract = abstract_tag.text.strip().replace('Abstract:', '')
                    st.markdown(f"""
                        <div class="recommended-paper">
                            <h3 class="title-style">{title}</h3>
                            <p class="abstract-style">摘要：{abstract}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    processed_title = preprocess_title(title)
                    input_titles.append(processed_title)
                    input_abstracts.append(abstract)
                    count += 1
                    input_text = processed_title + " " + abstract
                    input_embedding = sentence_encoder_layer([input_text]).numpy()[0]
                    viewed_embeddings.append(input_embedding)
                if count == 5:
                    break

            if count == 0:
                st.write("没有找到任何 arxiv 论文记录")
            else:
                if algorithm == "KNN":
                    avg_embedding = np.mean(viewed_embeddings, axis=0)
                    recommended_papers = get_recommendations_knn(avg_embedding, nn_model, df, embeddings, input_titles)
                elif algorithm == "TF-IDF":
                    avg_text = " ".join([title + " " + abstract for title, abstract in zip(input_titles, input_abstracts)])
                    recommended_papers = get_recommendations_tfidf(avg_text, vectorizer, tfidf_matrix, df, input_titles)
                elif algorithm == "FAISS":
                    avg_embedding = np.mean(viewed_embeddings, axis=0)
                    recommended_papers = get_recommendations_faiss(avg_embedding, index, df, input_titles)
                
                if recommended_papers:
                    st.markdown(f"""
                    <div class="recommended-paper">
                            <h3 class="title-style" style="color: brown;">推荐的论文列表：</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    for paper in recommended_papers:
                        # change the color of the title to be red
                        st.markdown(f"""
                        <div class="recommended-paper">
                            <h3 class="title-style" style="color: brown;">{paper['title']} ({paper['year']})</h3>
                            <p class="abstract-style">摘要：{paper['abstract']}</p>
                            <p class="abstract-style">相似度：{paper['similarity_score']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    fig, ax = plt.subplots()
                    titles = [paper['title'] for paper in recommended_papers]
                    similarities = [paper['similarity_score'] for paper in recommended_papers]
                    ax.barh(titles, similarities, color='skyblue')
                    ax.set_xlabel('Similarity')
                    ax.set_xlim(0, 1)
                    st.pyplot(fig)
                else:
                    st.write("没有推荐的论文")
        else:
            st.write("没有找到任何浏览记录")
