import copy as cp
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# Plot a confusion matrix for the result of a prediction
# param mode can be 'absolute' or 'relative'
def plot_confusion_matrix(actual_classes, predicted_classes, sorted_labels, mode):

    if mode not in ['absolute','relative']:
        raise Exception(f"mode {mode} is not supported. Use 'absolute' or 'relative'.")
    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    matrix_normalized = matrix / matrix.sum(axis=1, keepdims=True)
    matrix_normalized = np.round_(matrix_normalized, decimals=5)
    
    if mode == 'absolute':
        plt.figure(figsize=(12.8,6))
        sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    else:
        plt.figure(figsize=(16,10))
        sns.heatmap(matrix_normalized, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix mit relativen Werten')

    plt.show()

def cross_val_predict(model, kfold, X, y):

    model_ = cp.deepcopy(model)
    
    no_classes = len(np.unique(y))
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba