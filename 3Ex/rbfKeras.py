import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import keras
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('./breast_cancer.csv')
for i in 'id	diagnosis	radius_mean	texture_mean	perimeter_mean	area_mean	smoothness_mean	compactness_mean	concavity_mean	concave points_mean	symmetry_mean	fractal_dimension_mean	radius_se	texture_se	perimeter_se	area_se	smoothness_se	compactness_se	concavity_se	concave points_se	symmetry_se	fractal_dimension_se	radius_worst	texture_worst	perimeter_worst	area_worst	smoothness_worst	compactness_worst	concavity_worst	concave points_worst	symmetry_worst	fractal_dimension_worst'.split(' '):
    print(i.rstrip('    '))
# df.columns = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
#               'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
#               'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#               'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#               'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
#               'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
#               'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
#               'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# X = df.iloc[:, 2:].values
# y = df.iloc[:, 1].values
# y = np.where(y == 'M', 1, 0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# class RBFLayer(Dense):
#     def __init__(self, units, gamma, **kwargs):
#         self.units = units
#         self.gamma = K.variable(gamma)
#         super(RBFLayer, self).__init__(units, **kwargs)

#     def build(self, input_shape):
#         self.mu = self.add_weight(name='mu',
#                                   shape=(int(input_shape[1]), self.units),
#                                   initializer=RandomUniform(minval=-1, maxval=1),
#                                   trainable=True)
#         super(RBFLayer, self).build(input_shape)

#     def call(self, inputs):
#         diff = K.expand_dims(inputs) - self.mu
#         l2 = K.sum(K.pow(diff, 2), axis=1)
#         res = K.exp(-1 * self.gamma * l2)
#         return res

# model = Sequential()
# model.add(RBFLayer(10, 0.5, input_shape=(10,)))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)

# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.initializers import RandomUniform
# import numpy as np

# # Create some random input and output data
# x_train = np.random.random((100, 10))
# y_train = np.random.randint(2, size=(100, 1))

# class RBFLayer(Dense):
#     def __init__(self, units, gamma, **kwargs):
#         self.units = units
#         self.gamma = K.variable(gamma)
#         super(RBFLayer, self).__init__(units, **kwargs)

#     def build(self, input_shape):
#         self.mu = self.add_weight(name='mu',
#                                   shape=(int(input_shape[1]), self.units),
#                                   initializer=RandomUniform(minval=-1, maxval=1),
#                                   trainable=True)
#         super(RBFLayer, self).build(input_shape)

#     def call(self, inputs):
#         diff = K.expand_dims(inputs) - self.mu
#         l2 = K.sum(K.pow(diff, 2), axis=1)
#         res = K.exp(-1 * self.gamma * l2)
#         return res

# model = Sequential()
# model.add(RBFLayer(10, 0.5, input_shape=(10,)))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=100)

