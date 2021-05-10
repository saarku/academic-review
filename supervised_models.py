from sklearn.neural_network import MLPRegressor
from feature_builder import FeatureBuilder
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
import joblib
from scipy.stats import kendalltau
import numpy as np
from svm_rank import SVMRank

'''
TODO:
0. SVM rank
1. Feature normalization
2. Score normalization
3. Why do we have negative kendall?
4. Saving models
'''

data_dir = '/home/skuzi2/iclr17_dataset'
topics_dir = data_dir + '/lda_vectors/'
model_name = data_dir + '/models/'
test_dimensions = [1, 2, 3, 5, 6]
modes = ['pos', 'neg']
dimension_features = {'1': modes, '2': modes, '3': modes, '5': modes, '6': modes, 'all': ['neu']}
topic_model_dims = [5]
num_paragraphs = [1]

unigrams_flag = False
feature_comb_flag = True

builder = FeatureBuilder(data_dir)

for dim in test_dimensions:
    model_name = 'dim.' + str(dim)
    all_features_train, all_features_test = [], []
    y_train, y_test = [], []

    if unigrams_flag:
        x_unigram_train, y_train, x_unigram_test, y_test = builder.build_unigram_features(dim)
        all_features_train.append(x_unigram_train)
        all_features_test.append(x_unigram_test)
        model_name += '.uni'

    for topics in topic_model_dims:
        for para in num_paragraphs:
            for dim_feat in dimension_features:
                for mode in dimension_features[dim_feat]:
                    vec_dir = topics_dir
                    vec_dir += '{}_topics/dim.{}.mod.{}.para.{}.num.{}'.format(topics, dim_feat, mode, para, topics)
                    output = builder.build_topic_features(dim, vec_dir + '.train', vec_dir + '.test.val', para)
                    x_topics_train, y_train, x_topics_test, y_test = output[0], output[1], output[2], output[3]
                    all_features_train.append(x_topics_train)
                    all_features_test.append(x_topics_test)
    model_name += '.topics'

    if feature_comb_flag:
        train_features = sp.hstack(tuple(all_features_train), format='csr')
        test_features = sp.hstack(tuple(all_features_test), format='csr')
        #clf = MLPRegressor(solver='sgd', max_iter=500, verbose=False).fit(train_features, y_train)
        clf = SVMRank()
        clf.fit(train_features, y_train, model_name, 0.01)
        grades = clf.predict(test_features)
    else:
        grades = np.zeros((all_features_test[0].shape[0], 1))
        counter = 0
        for i in range(len(all_features_train)):
            counter += 1
            #clf = MLPRegressor(solver='sgd', max_iter=500, verbose=False).fit(all_features_train[i], y_train)
            clf = SVMRank()
            clf.fit(all_features_train[i], y_train, model_name, 0.01)
            aspect_grades = clf.predict(all_features_test[i])
            aspect_grades = np.reshape(aspect_grades, (-1, 1))
            grades += aspect_grades
        grades /= counter

    error = sqrt(mean_squared_error(y_test, grades))
    kendall, _ = kendalltau(y_test, grades)
    print(str(dim) + ',' + str(error) + ',' + str(kendall))
    #joblib.dump(clf, model_name + '.joblib')