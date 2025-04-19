import re
from collections import defaultdict

# Step 1: Preprocess documents
def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())  # Tokenize and lowercase

# Documents
documents = {
    1: "best of luck tycs students for your practical examination.",
    2: "tycs students please carry your journal at the time of practical examination."
}

# Step 2: Build inverted index
inverted_index = defaultdict(set)

for doc_id, text in documents.items():
    words = preprocess(text)
    for word in words:
        inverted_index[word].add(doc_id)

# Step 3: Display inverted index
print("Inverted Index:")
for term in sorted(inverted_index):
    print(f"{term}: {sorted(inverted_index[term])}")

# Step 4: Document Retrieval
def retrieve_docs(query, index):
    query_terms = preprocess(query)
    result_sets = [index[term] for term in query_terms if term in index]
    
    # Get documents containing all query terms (intersection)
    if result_sets:
        result = set.intersection(*result_sets)
    else:
        result = set()
    return result

# Query: "tycs journal"
query = "tycs journal"
retrieved_docs = retrieve_docs(query, inverted_index)

print(f"\nQuery: \"{query}\"")
print("Matching Document(s):", sorted(retrieved_docs) if retrieved_docs else "No match found.")