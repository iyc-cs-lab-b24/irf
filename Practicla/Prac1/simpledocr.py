import re
from collections import defaultdict

class DocumentRetrievalSystem:
    def __init__(self):
        self.index = defaultdict(list)
        self.documents = []

    def add_document(self, document):
        doc_id = len(self.documents)
        self.documents.append(document)
        terms = self.tokenize(document)
        for term in terms:
            self.index[term].append(doc_id)

    def search(self, query):
        query_terms = self.tokenize(query)
        result_docs = set(self.index[query_terms[0]]) if query_terms and query_terms[0] in self.index else set()

        for term in query_terms[1:]:
            if term in self.index:
                result_docs.intersection_update(self.index[term])
            else:
                result_docs.clear()

        if result_docs:
            return [self.documents[doc_id] for doc_id in result_docs]
        else:
            return []  

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())  

if __name__ == "__main__":
    retrieval_system = DocumentRetrievalSystem()  

    retrieval_system.add_document("This is the first document about Python")
    retrieval_system.add_document("Python is a popular programming language")
    retrieval_system.add_document("Document retrieval systems are important for information retrieval.")

    query = "python" 
    results = retrieval_system.search(query)  

    if results:
        print("Search results for '{}':".format(query))  
        for result in results:
            print("-", result)
    else:
        print("No results found for '{}'.".format(query))