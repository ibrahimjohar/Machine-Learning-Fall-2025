import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def chi_square_dist(x,y):
    return np.sum(((x - y) ** 2) / (x + y))

#custom prediction function
def knn_prediction(X_train, y_train, testing_sample, k=3):
    distances = []
    
    #calculate the distance from testing_sample to each of the training samples
    for i in range(len(X_train)):
        d = chi_square_dist(testing_sample, X_train[i])
        distances.append((d, y_train[i]))       #storing (distance,label)
        
    #sort ascendingly the neigbors by distance
    distances.sort(key=lambda x:x[0])
    
    #selecting the top k NNs
    neighbors = distances[:k]
    
    targets = [label for _, label in neighbors]
    
    #majority voting
    most_common_target = Counter(targets).most_common(1)[0][0]
    
    return most_common_target

#ANOTHER METHOD
def knn(X_train, y_train, X_test, k=3):
    preds = []
    
    for x in X_test:
        dist = np.sqrt(np.sum((X_train - x)**2, axis=1))
        k_index = np.argsort(dist)[:k]
        k_labels = y_train[k_index]
        preds.append(Counter(k_labels).most_common(1)[0][0])
    
    return np.array(preds)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

y_pred = knn(X_train, y_train, X_test, k=5)
print(f"acc: {accuracy_score(y_test, y_pred)}") 
    
