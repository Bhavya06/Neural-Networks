 #!/usr/bin/env python -W ignore::DeprecationWarning
import mlrose
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score

l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l

eyes = generateColumns(1, 12)

import pandas as pd
df = pd.read_csv('Eyes.csv')
# print("hello")
X = df[eyes]
y = df['truth_value']   #the actual class labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Data Normalization
from sklearn.preprocessing import StandardScaler as SC
sc = SC()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)
#not scaling y since it's already 0s and 1s

import numpy as np
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
#converting all the scaled data to numpy arrays
'''
nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                 algorithm ='random_hill_climb', 
                                 max_iters = 1000, bias = True, is_classifier = True, 
                                 learning_rate = 0.0001, early_stopping = True, 
                                 clip_max = 5, max_attempts = 100, random_state = 3)

nn_model1.fit(X_train_scaled, y_train)

# print('Works so far')

y_train_pred = nn_model1.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print(y_train_accuracy)

y_test_pred = nn_model1.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print(y_test_accuracy)
'''

nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                 algorithm = 'gradient_descent', 
                                 max_iters = 1000, bias = True, is_classifier = True, 
                                 learning_rate = 0.0001, early_stopping = True, 
                                 clip_max = 5, max_attempts = 100, random_state = 3)

nn_model2.fit(X_train_scaled, y_train)

y_train_pred = nn_model2.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print(y_train_accuracy)

y_test_pred = nn_model2.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print(y_test_accuracy)
