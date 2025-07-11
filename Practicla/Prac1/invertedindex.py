import nltk  # Import NLTK to download stopwords 
from nltk.corpus import stopwords  # Import stopwords from NLTK 

# Define the documents 
document1 = "The quick brown fox jumped over the lazy dog" 
document2 = "The lazy dog slept in the sun" 

# Get the stopwords for English language from NLTK 
nltk.download('stopwords') 
stopWords = stopwords.words('english') 

# Step 1: Tokenize the documents 
# Convert each document to lowercase and split it into words 
tokens1 = document1.lower().split() 
tokens2 = document2.lower().split() 

# Combine the tokens into a list of unique terms 
terms = list(set(tokens1 + tokens2)) 

# Step 2: Build the inverted index 
# Create an empty dictionary to store the inverted index 
# and dictionaries to store number of occurrences 
inverted_index = {} 
occ_num_doc1 = {} 
occ_num_doc2 = {} 

# For each term, find the documents that contain it 
for term in terms: 
    if term in stopWords: 
        continue 
    documents = [] 
    if term in tokens1: 
        documents.append("Document 1") 
        occ_num_doc1[term] = tokens1.count(term) 
    if term in tokens2: 
        documents.append("Document 2") 
        occ_num_doc2[term] = tokens2.count(term) 

    inverted_index[term] = documents 

# Step 3: Print the inverted index 
for term, documents in inverted_index.items(): 
    print(term, "->", end=" ") 
    for doc in documents: 
        if doc == "Document 1": 
            print(f"{doc} ({occ_num_doc1.get(term, 0)}),", end=" ") 
        else: 
            print(f"{doc} ({occ_num_doc2.get(term, 0)}),", end=" ") 
    print()