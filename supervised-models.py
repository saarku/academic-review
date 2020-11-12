from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor
from feature_builder import FeatureBuilder
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

data_dir = '/home/skuzi2/iclr17_dataset'
dimensions = [1, 2, 3, 5, 6]

builder = FeatureBuilder(data_dir)
topics_dir = '/home/skuzi2/iclr17_dataset/lda_models/100_topics/100_topics.'

for dim in dimensions:
    x_train, y_train, x_test, y_test = builder.build_topic_features(dim, topics_dir + 'train', topics_dir + 'test')
    y_train = np.asarray(y_train, dtype=float)
    clf = MLPRegressor(solver='sgd').fit(x_train, y_train)
    grades = clf.predict(x_test)
    error = sqrt(mean_squared_error(y_test, grades))
    print(str(dim) + ',' + str(error))

# test_dataset = sp.hstack(tuple([x_test_counts, x_test_tf_idf]), format='csr')


'''
for dim in dimensions:
    builder = FeatureBuilder(data_dir)
    x_train, y_train, x_test, y_test = builder.build_unigram_features(dim)
    clf = LinearRegression().fit(x_train, y_train)
    grades = clf.predict(x_test)
    error = sqrt(mean_squared_error(y_test, grades))
    print(str(dim) + ',' + str(error))

builder = FeatureBuilder(data_dir)
for dim in dimensions:
    x_train, y_train, x_test, y_test = builder.build_unigram_features(dim)
    grades = [np.mean(y_train)]*len(y_test)
    error = sqrt(mean_squared_error(y_test, grades))
    print(str(dim) + ',' + str(error))
'''
