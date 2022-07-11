import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size = 0.2, random_state=3)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = 'random_hill_climb', max_iters = 1000,
                                 bias = True, is_classifier = True, learning_rate = 0.0001,
                                 early_stopping = True, clip_max = 5, max_attempts = 100,
				                 random_state = 3)

nn_model1.fit(X_train_scaled, y_train_hot)

y_train_pred = nn_model1.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print('training accuracy: ', y_train_accuracy)

y_test_pred = nn_model1.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print('test accuracy: ', y_test_accuracy)