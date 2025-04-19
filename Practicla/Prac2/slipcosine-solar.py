from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Documents
documents = [
    "The sun is the star at the center of the solar system.",
    "She wore a beautiful dress to the party last night.",
    "The book on the table caught my attention immediately."
]

# Query
query = "solar system"

# Combine documents and query
corpus = documents + [query]

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

# Separate the query vector (last row)
query_vector = tfidf_matrix[-1]
document_vectors = tfidf_matrix[:-1]

# Compute cosine similarities
cosine_similarities = cosine_similarity(query_vector, document_vectors).flatten()

# Print the similarity between query and each document
print("Cosine Similarities:")
for i, score in enumerate(cosine_similarities):
    print(f"Query vs Document {i+1}: {score:.4f}")

# Find the most similar document
most_similar_doc_index = cosine_similarities.argmax()
print(f"\nThe query is most similar to Document {most_similar_doc_index + 1}")