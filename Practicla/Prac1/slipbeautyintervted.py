from collections import defaultdict

# Step 1: Define documents
documents = {
    1: "today is a beautiful and a sunny day",
    2: "it was a cloudy day"
}

# Step 2: Function to build inverted index
def build_inverted_index(docs):
    index = defaultdict(set)
    for doc_id, text in docs.items():
        words = text.lower().split()
        for word in words:
            index[word].add(doc_id)
    return index

# Step 3: Build the inverted index
inverted_index = build_inverted_index(documents)

# Optional: Display the inverted index
print("🔎 Inverted Index:")
for word, doc_ids in inverted_index.items():
    print(f"{word}: {sorted(doc_ids)}")

# Step 4: Search for query terms
query_terms = ["beautiful", "day"]

# Step 5: Retrieve documents containing all query terms
matching_docs = set.intersection(*(inverted_index.get(term, set()) for term in query_terms))

# Step 6: Display results
print("\n📄 Documents containing all terms in query 'beautiful day':")
if matching_docs:
    for doc_id in sorted(matching_docs):
        print(f"Document {doc_id}: {documents[doc_id]}")
else:
    print("No matching documents found.")