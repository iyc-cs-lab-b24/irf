from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Step 1: Small dataset of 4 documents
docs = [
    "I love programming in Python.",
    "Machine learning is fun and exciting.",
    "I enjoy watching cricket and football.",
    "Sports like cricket and football are popular."
]

# Step 2: Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(docs)

# Step 3: Apply KMeans clustering (2 clusters)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Step 4: Print cluster labels
print("Cluster labels for each document:")
for i, label in enumerate(kmeans.labels_):
    print(f"Document {i+1}: Cluster {label}")

# Step 5: Print documents grouped by cluster
print("\nDocuments grouped by cluster:")
for cluster in range(2):
    print(f"\n--- Cluster {cluster} ---")
    for i in range(len(docs)):
        if kmeans.labels_[i] == cluster:
            print(docs[i])
