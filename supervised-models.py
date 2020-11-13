from sklearn.neural_network import MLPRegressor
from feature_builder import FeatureBuilder
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
import joblib
from scipy.stats import kendalltau

data_dir = '/home/skuzi2/iclr17_dataset'
dimensions = [1, 2, 3, 5, 6]
topic_model_dims = []
unigrams_flag = True

builder = FeatureBuilder(data_dir)
topics_dir = '/home/skuzi2/iclr17_dataset/lda_models/'
model_name = '/home/skuzi2/iclr17_dataset/models/'


for dim in dimensions:
    model_name += str(dim)
    all_features_train = []
    all_features_test = []
    if unigrams_flag:
        x_unigram_train, y_train, x_unigram_test, y_test = builder.build_unigram_features(dim)
        all_features_train.append(x_unigram_train)

        all_features_test.append(x_unigram_test)
        model_name += '.uni'

    for topic_num in topic_model_dims:
        topic_model_dir = topics_dir + '/' + str(topic_num) + '_topics/' + str(topic_num) + '_topics'
        x_topics_train, y_train, x_topics_test, y_test = builder.build_topic_features(dim,
                                                                                      topic_model_dir + '.train',
                                                                                      topic_model_dir + '.test')
        all_features_train.append(x_topics_train)

        all_features_test.append(x_topics_test)
        model_name += '.topic' + str(topic_num)

    train_features = sp.hstack(tuple(all_features_train), format='csr')
    test_features = sp.hstack(tuple(all_features_test), format='csr')
    clf = MLPRegressor(solver='sgd', max_iter=500, verbose=False).fit(train_features, y_train)
    grades = clf.predict(test_features)
    error = sqrt(mean_squared_error(y_test, grades))
    kendall, _ = kendalltau(y_test, grades)
    print(str(dim) + ',' + str(error) + ',' + str(kendall))
    joblib.dump(clf, model_name + '.joblib')


'''
#y_train = np.asarray(y_train, dtype=float)
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
