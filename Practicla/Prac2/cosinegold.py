from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Query and Document
query = ["gold silver truck"]
document = ["shipment of gold damaged in a gold fire"]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the documents and query
tfidf_matrix = vectorizer.fit_transform(query + document)

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Display the similarity score
print(f"Cosine Similarity between query and document: {cosine_sim[0]:.4f}")