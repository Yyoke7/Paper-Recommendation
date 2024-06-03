import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

@st.cache_resource
def get_knn_model(embeddings):
    y = np.arange(len(embeddings))
    nn = KNeighborsClassifier(n_neighbors=10)
    nn.fit(embeddings, y)
    return nn

@st.cache_resource
def load_model():
    return hub.KerasLayer(MODEL_URL, input_shape=[], dtype=tf.string, trainable=False, name="use")