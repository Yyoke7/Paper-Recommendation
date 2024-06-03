import warnings
warnings.filterwarnings('ignore')

import os
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pickle


FILE = './data/arxiv-metadata-oai-snapshot.json'

def get_data():
    with open(FILE) as f:
        for line in f:
            yield line


dataframe = {
    'title': [],
    'year': [],
    'abstract': []
}

data = get_data()
for i, paper in enumerate(data):
    paper = json.loads(paper)
    try:
        date = int(paper['update_date'].split('-')[0])
        if date > 2019:
            dataframe['title'].append(paper['title'])
            dataframe['year'].append(date)
            dataframe['abstract'].append(paper['abstract'])
    except: pass

df = pd.DataFrame(dataframe)

# Tensorflow Hub URL for Universal Sentence Encoder
MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

# KerasLayer
sentence_encoder_layer = hub.KerasLayer(MODEL_URL,
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=False,
                                        name="use")

abstracts = df["abstract"].to_list()

# Setup for embeddings computation
embeddings = []
batch_size = 3000
num_batches = len(abstracts) // batch_size

# Compute Embeddings in batches
for i in range(num_batches):
    batch_abstracts = abstracts[i*batch_size : (i+1)*batch_size]
    batch_embeddings = sentence_encoder_layer(batch_abstracts)
    embeddings.extend(batch_embeddings.numpy())

# Embeddings for remaining abstracts
remaining_abstracts = abstracts[num_batches*batch_size:]
if len(remaining_abstracts) > 0:
    remaining_embeddings = sentence_encoder_layer(remaining_abstracts)
    embeddings.extend(remaining_embeddings.numpy())
    
embeddings = np.array(embeddings)
# Save embeddings in ./data/embeddings.pkl
with open('./data/embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

print("Embeddings saved successfully!")