from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans 

# Define the documents
documents = [
    "Cats are known for their agility and grace",  # cat doc1 
    "Dogs are often called ‘man’s best friend’.",  # dog doc1 
    "Some dogs are trained to assist people with disabilities.",  # dog doc2 
    "The sun rises in the east and sets in the west.",  # sun doc1 
    "Many cats enjoy climbing trees and chasing toys.",  # cat doc2 
] 

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer(stop_words='english') 

# Learn vocabulary and idf from training set
X = vectorizer.fit_transform(documents) 

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X) 

# Print cluster labels for each document
print(kmeans.labels_)

