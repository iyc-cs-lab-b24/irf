from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Corpus
documents = [
    "The sun is the star at the center of the solar system.",
    "She wore a beautiful dress to the party last night.",
    "The book on the table caught my attention immediately."
]

# Query
query = ["solar system"]

# Combine query and documents
corpus = documents + query

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

# Cosine Similarity of query with each document
cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Display Results
for idx, score in enumerate(cosine_similarities):
    print(f"Similarity with Document {idx+1}: {score:.4f}")

# Rank documents
most_similar_doc = np.argmax(cosine_similarities) + 1
print(f"\nThe query is most similar to Document {most_similar_doc}")