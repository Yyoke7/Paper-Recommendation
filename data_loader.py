import json
import pickle
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path):
    dataframe = {'title': [], 'year': [], 'abstract': []}
    with open(file_path) as f:
        for line in f:
            paper = json.loads(line)
            try:
                date = int(paper['update_date'].split('-')[0])
                if date > 2019:
                    dataframe['title'].append(paper['title'])
                    dataframe['year'].append(date)
                    dataframe['abstract'].append(paper['abstract'])
            except:
                pass
    return pd.DataFrame(dataframe)

@st.cache_data
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)