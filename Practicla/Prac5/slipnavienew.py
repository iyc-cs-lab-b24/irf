from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load a subset of 20newsgroups dataset with 2 categories for binary classification
categories = ['rec.autos', 'rec.sport.baseball']  # Consider one as positive, one as negative
train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# Create a pipeline: TF-IDF vectorizer + Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(train.data, train.target)

# Predict on test data
predicted = model.predict(test.data)

# Evaluate the model
print("Accuracy:", accuracy_score(test.target, predicted))
print("\nClassification Report:\n", classification_report(test.target, predicted, target_names=categories))
