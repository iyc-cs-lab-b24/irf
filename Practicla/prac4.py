from sklearn.metrics import average_precision_score 

# Binary Prediction
y_true = [0, 1, 1, 0, 1, 1] 

# Model's estimation score
y_scores = [0.1, 0.4, 0.35, 0.8, 0.65, 0.9] 

# Calculate the average precision-recall score
average_precision = average_precision_score(y_true, y_scores) 

# Print the average precision-recall score
print(f'Average precision-recall score: {average_precision}')
