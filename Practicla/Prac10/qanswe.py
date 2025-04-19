import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
data = {
    "What is the capital of France?": "Paris is the capital of France.",
    "Who painted the Mona Lisa?": "Leonardo da Vinci painted the Mona Lisa.",
    "When did World War II end?": "World War II ended in 1945."
}

questions = list(data.keys())
answers = list(data.values())

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(questions + answers)

def get_answer(question):
    # Vectorize the question
    question_vector = vectorizer.transform([question])

    # Calculate cosine similarity with question bank
    similarity_scores = cosine_similarity(question_vector, vectorizer.transform(questions))

    # Get the most similar question
    most_similar_index = similarity_scores.argmax()

    # Extract the answer
    return answers[most_similar_index]

# Example usage
question = "which war ended in 1945?"
answer = get_answer(question)
print(answer)
