import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

def generate_dataset():
    # Example dataset
    data = [
        {"query": "query1", "document": "doc1", "relevance": 3},
        {"query": "query1", "document": "doc2", "relevance": 2},
        {"query": "query1", "document": "doc3", "relevance": 1},
        {"query": "query2", "document": "doc4", "relevance": 3},
        {"query": "query2", "document": "doc5", "relevance": 1},
        {"query": "query2", "document": "doc6", "relevance": 2},
    ]

    # Convert to feature vectors and labels
    X = np.array([[3, 2, 1], [2, 1, 0], [0, 1, 2], [1, 2, 0], [2, 1, 3], [1, 0, 2]])
    y = np.array([1, 0, 0, 1, 0, 1])
    
    return X, y

def train_rank_svm(X_train, y_train):
    # Train RankSVM model
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Predict relevance scores for the test set
    y_pred = model.predict(X_test)

    # Calculate NDCG score (Normalized Discounted Cumulative Gain)
    ndcg = ndcg_score([y_test], [y_pred])
    print(f"NDCG Score: {ndcg:.4f}")

def main():
    # Prepare data
    X, y = generate_dataset()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RankSVM model
    model = train_rank_svm(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
