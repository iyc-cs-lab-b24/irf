from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define the query and document
query = "python programming"
document = "Document about python programming language and data analysis."

# Step 2: Create a TF-IDF Vectorizer and apply it to both query and document
vectorizer = TfidfVectorizer()

# Combine query and document into a single list
corpus = [query, document]

# Fit and transform the data into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(corpus)

# Step 3: Calculate cosine similarity between the query (index 0) and the document (index 1)
cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Step 4: Print the cosine similarity
print(f"Cosine Similarity between query and document: {cos_sim[0][0]:.4f}")