import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_title

def get_recommendations_knn(embedding, nn_model, df, embeddings, viewed_titles):
    distances, indices = nn_model.kneighbors([embedding], n_neighbors=len(df))
    recommended_papers = []
    for i, index in enumerate(indices[0]):
        title = preprocess_title(df['title'][index])
        if title not in viewed_titles:
            similarity_score = 1 - distances[0][i]
            recommended_papers.append({
                'title': df['title'][index],
                'year': df['year'][index],
                'similarity_score': similarity_score,
                'abstract': df['abstract'][index]
            })
        if len(recommended_papers) == 6:
            break
    return recommended_papers

def get_recommendations_tfidf(input_text, vectorizer, tfidf_matrix, df, viewed_titles):
    input_tfidf = vectorizer.transform([input_text])
    cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    recommended_papers = []
    for i in related_docs_indices:
        title = preprocess_title(df['title'][i])
        if title not in viewed_titles:
            recommended_papers.append({
                'title': df['title'][i],
                'year': df['year'][i],
                'similarity_score': cosine_similarities[i],
                'abstract': df['abstract'][i]
            })
        if len(recommended_papers) == 6:
            break
    return recommended_papers

def his_get_recommendations(viewed_embeddings, nn_model, df, embeddings, viewed_titles, num_recommendations=6):
    distances = []
    for i in range(len(embeddings)):
        distance = np.linalg.norm(viewed_embeddings - embeddings[i])
        distances.append(distance)
    distances = np.array(distances)
    indices = np.argsort(distances)
    recommended_papers = []
    for i in indices:
        title = preprocess_title(df['title'][i])
        if title not in viewed_titles:
            print(title)
            print(viewed_titles)
            similarity_score = 1 - distances[i]
            recommended_papers.append({
                'title': df['title'][i],
                'year': df['year'][i],
                'similarity_score': similarity_score,
                'abstract': df['abstract'][i]
            })
        if len(recommended_papers) == num_recommendations:
            break
    return recommended_papers

def get_recommendations_faiss(query_embedding, index, df, viewed_titles, k=10):
    distances, indices = index.search(np.array([query_embedding]), k)
    recommended_papers = []
    for i, index in enumerate(indices[0]):
        title = preprocess_title(df['title'][index])
        if title not in viewed_titles:
            similarity_score = 1 - distances[0][i]
            recommended_papers.append({
                'title': df['title'][index],
                'year': df['year'][index],
                'similarity_score': similarity_score,
                'abstract': df['abstract'][index]
            })
        if len(recommended_papers) == 6:
            break
    return recommended_papers