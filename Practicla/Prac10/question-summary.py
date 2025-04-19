import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    
    word_frequencies = {}
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            word = ps.stem(word)
            if word not in stop_words:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
    
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)
    
    return word_frequencies, sentences

def calculate_sentence_scores(sentences, word_frequencies):
    sentence_scores = {}
    for sentence in sentences:
        sentence_word_count = len(word_tokenize(sentence))
        sentence_score = 0
        for word in word_tokenize(sentence):
            if word in word_frequencies:
                sentence_score += word_frequencies[word]
        if sentence_word_count > 0:
            sentence_scores[sentence] = sentence_score / sentence_word_count
    return sentence_scores

def generate_summary(text, num_sentences):
    word_frequencies, sentences = preprocess_text(text)
    sentence_scores = calculate_sentence_scores(sentences, word_frequencies)
    sorted_sentence_scores = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary_sentences = [sentence[0] for sentence in sorted_sentence_scores[:num_sentences]]
    summary = ' '.join(summary_sentences)
    return summary


text = """
Natural language processing (NLP) is a field of computer science, artificial intelligence, and computational linguistics concerned with the interactions between computers and human (natural) languages. 
As such, NLP is related to the area of human–computer interaction. 
Many challenges in NLP involve natural language understanding, that is, enabling computers to derive meaning from human or natural language input, and others involve natural language generation.
"""
summary = generate_summary(text, 2)
print(summary)