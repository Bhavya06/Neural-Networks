 #!/usr/bin/env python -W ignore::DeprecationWarning
import mlrose
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np

startTime = datetime.now()
print("\n\n\nStarting the execution now:\n")

l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l
eyes = generateColumns(1, 12)
import pandas as pd
df = pd.read_csv('Eyes.csv')
first_column = df.columns[0]
df = df.drop([first_column],axis = 1)
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
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
#converting all the scaled data to numpy arrays

iters = 100
initial_acc = 50
acc = 0



def gradDesc(iterations):
    nn_model = mlrose.NeuralNetwork(hidden_nodes = [4], activation = 'relu', 
                                    algorithm = 'gradient_descent', 
                                    max_iters = iterations, bias = True, is_classifier = True, 
                                    learning_rate = 0.0001, early_stopping = True, 
                                    clip_max = 5, max_attempts = 100, random_state = 3)

    nn_model.fit(X_train_scaled, y_train)

    # y_train_pred = nn_model.predict(X_train_scaled)
    # y_train_accuracy = accuracy_score(y_train, y_train_pred)
    # print("The Training accuracy is: ",y_train_accuracy*100,"%")

    y_test_pred = nn_model.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    return (y_test_pred, y_test_accuracy)
# acc = y_test_accuracy*100

for i in range(1, 10):
    y_test_accuracy = gradDesc(iters)
    acc = y_test_accuracy[1] * 100
    print(acc)
    if (acc < initial_acc ):
        iters = iters + 1000
    else:
        print("Higher accuracy found: ", acc, "With iterations: ", iters)
        iters = iters + 1000
    if (acc > 95):
        break



# print("The Testing accuracy is: ",y_test_accuracy[1]*100,"%")

print("\n")
print("Execution time in seconds = ", datetime.now() - startTime)
