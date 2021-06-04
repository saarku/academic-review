from sklearn.neural_network import MLPRegressor
from feature_builder import FeatureBuilder
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
import joblib
from scipy.stats import kendalltau, pearsonr
import numpy as np
from svm_rank import SVMRank
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
import sys
import random
import time
import os


def learn_model(algorithm, features, labels, model_dir):
    if algorithm == 'regression':
        clf = LinearRegression()
        clf.fit(features, labels)
        joblib.dump(clf, model_dir)
    else:
        clf = SVMRank()
        clf.fit(features, labels, model_dir, 0.01)
    return clf


def run_sum_comb_method(all_train_features, train_labels, all_test_features, test_labels, algorithm, method):
    grades = np.zeros((len(test_labels), 1), dtype=float)
    for i in range(len(all_train_features)):
        train_features, test_features = all_train_features[i], all_test_features[i]
        train_features, test_features = train_features.todense(), test_features.todense()
        transformer = MinMaxScaler()
        transformer.fit(train_features)
        train_features, test_features = transformer.transform(train_features), transformer.transform(test_features)

        temp_model_dir = 'val.' + str(time.time())
        clf = learn_model(algorithm, train_features, train_labels, temp_model_dir)
        aspect_grades = clf.predict(test_features)
        aspect_grades = np.reshape(aspect_grades, (-1, 1))
        os.system('rm -rf ' + temp_model_dir)

        if method == 'comb_rank':
            aspect_grades = FeatureBuilder.grades_to_ranks(aspect_grades)

        if algorithm == 'ranking' and method == 'comb_sum':
            grades += softmax(aspect_grades)
        else:
            grades += aspect_grades

    grades /= float(len(all_train_features))
    return grades


def single_experiment(test_dimensions, data_dir, unigrams_flag, combination_method, num_topic, num_paragraphs,
                      dimension_features, algorithm, kl_flag, model_type):

    output_performance = ''
    builder = FeatureBuilder(data_dir)
    topics_dir, models_dir = data_dir + '/lda_vectors_{}/'.format(model_type), data_dir + '/models/'
    modes = set()
    for i in dimension_features: modes = modes.union(set(dimension_features[i]))
    modes = '_'.join(sorted([str(i) for i in modes]))
    paragraph_id = '_'.join([str(i) for i in num_paragraphs])

    for dim in test_dimensions:
        model_dir = models_dir + 'dim.' + str(dim) + '.algo.' + algorithm
        model_dir += '.topics.' + str(num_topic) + '.para.' + paragraph_id
        model_dir += '.mode.' + modes + '.kl.' + str(kl_flag).lower() + '.type.' + str(model_type).lower()
        model_dir += '.uni.' + str(unigrams_flag).lower() + '.comb.' + combination_method
        uni_features_train, uni_features_test, y_train, y_test = [], [], [], []

        if unigrams_flag:
            x_unigram_train, y_train, x_unigram_test, y_test = builder.build_unigram_features(dim)
            uni_features_train.append(x_unigram_train)
            uni_features_test.append(x_unigram_test)

        topic_model_train_features, topic_model_test_features = {}, {}
        topic_model_dims = [5, 15, 25] if num_topic == 'cv' else [num_topic]
        for topics in topic_model_dims:
            topic_model_train_features[topics], topic_model_test_features[topics] = [], []
            for para in num_paragraphs:
                for dim_feat in dimension_features:
                    for mode in dimension_features[dim_feat]:
                        vec_dir = topics_dir
                        vec_dir += '{}_topics/dim.{}.mod.{}.para.{}.num.{}'.format(topics, dim_feat, mode, para, topics)
                        if kl_flag == 'kl' or kl_flag == 'normkl': vec_dir += '.kl'
                        norm = True if kl_flag == 'normkl' else False
                        output = builder.build_topic_features(dim, vec_dir + '.train', vec_dir + '.test.val', para,
                                                              norm=norm)
                        x_topics_train, y_train, x_topics_test, y_test = output[0], output[1], output[2], output[3]
                        topic_model_train_features[topics].append(x_topics_train)
                        topic_model_test_features[topics].append(x_topics_test)

        if combination_method == 'feature_comb':
            optimal_num, optimal_kendall = 0, -1
            for topic_num in topic_model_train_features:
                train_features = tuple(topic_model_train_features[topic_num] + uni_features_train)
                train_features = sp.hstack(train_features, format='csr')
                all_train_ids = list(range(len(y_train)))
                random.shuffle(all_train_ids)
                val_split = int(len(all_train_ids) * 0.15)
                validation_ids, small_train_ids = all_train_ids[:val_split], all_train_ids[val_split:]
                validation_labels = [y_train[i] for i in range(len(y_train)) if i in validation_ids]
                small_train_labels = [y_train[i] for i in range(len(y_train)) if i in small_train_ids]

                transformer = MinMaxScaler()
                train_features = train_features.todense()
                transformer.fit(train_features)
                small_train_features = train_features[small_train_ids, :]
                validation_features = train_features[validation_ids, :]
                small_train_features = transformer.transform(small_train_features)
                validation_features = transformer.transform(validation_features)

                temp_model_dir = 'val.comb.feature.' + str(time.time())
                clf = learn_model(algorithm, small_train_features, small_train_labels, temp_model_dir)
                val_grades = clf.predict(validation_features)
                kendall, _ = kendalltau(validation_labels, np.reshape(val_grades, (-1, 1)))
                os.system('rm ' + temp_model_dir)

                if kendall > optimal_kendall:
                    optimal_kendall = kendall
                    optimal_num = topic_num

            train_features = sp.hstack(tuple(topic_model_train_features[optimal_num] + uni_features_train), format='csr')
            test_features = sp.hstack(tuple(topic_model_test_features[optimal_num] + uni_features_test), format='csr')
            transformer = MinMaxScaler()
            transformer.fit(train_features)
            train_features = transformer.transform(train_features)
            test_features = transformer.transform(test_features)
            clf = learn_model(algorithm, train_features, y_train, model_dir + '.model')
            grades = clf.predict(test_features)
            open(model_dir + '.predict', 'w').write('\n'.join([str(grades[i,0]) for i in range(grades.shape[0])]))

        else:
            optimal_num, optimal_kendall = 0, -1
            for topic_num in topic_model_train_features:
                train_features = topic_model_train_features[topic_num] + uni_features_train
                all_train_ids = list(range(len(y_train)))
                random.shuffle(all_train_ids)
                val_split = int(len(all_train_ids) * 0.15)
                validation_ids, small_train_ids = all_train_ids[:val_split], all_train_ids[val_split:]
                validation_labels = [y_train[i] for i in range(len(y_train)) if i in validation_ids]
                small_train_labels = [y_train[i] for i in range(len(y_train)) if i in small_train_ids]

                small_train_features, validation_features = [], []
                for features in train_features:
                    small_train_features.append(features[small_train_ids, :])
                    validation_features.append(features[validation_ids, :])
                val_grades = run_sum_comb_method(small_train_features, small_train_labels, validation_features,
                                                 validation_labels, algorithm, combination_method)
                kendall, _ = kendalltau(validation_labels, np.reshape(val_grades, (-1, 1)))
                if kendall > optimal_kendall:
                    optimal_kendall = kendall
                    optimal_num = topic_num

            train_features = topic_model_train_features[optimal_num] + uni_features_train
            test_features = topic_model_test_features[optimal_num] + uni_features_test
            grades = run_sum_comb_method(train_features, y_train, test_features, y_test, algorithm, combination_method)

            open(model_dir + '.comb.' + combination_method + '.predict', 'w').write(
                '\n'.join([str(grades[i, 0]) for i in range(grades.shape[0])]))

        error = sqrt(mean_squared_error(y_test, grades))
        kendall, _ = kendalltau(y_test, grades)
        pearson, _ = pearsonr(y_test, np.reshape(grades, (1, -1)).tolist()[0])
        performance = '{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(dim, unigrams_flag, combination_method,
                                                                        num_topic, optimal_num, paragraph_id, algorithm,
                                                                        modes, kl_flag, model_type, error, kendall,
                                                                        pearson)
        output_performance += performance
    return output_performance


def run_experiments():
    data_name = sys.argv[1]
    topic_model_type = 'ovb'
    data_dir = '/home/skuzi2/{}_dataset'.format(data_name)
    test_dimensions = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]
    topic_model_dims = ['cv']

    modes, pos_modes, neg_modes = ['pos', 'neg'], ['pos'], ['neg']
    dimension_features, pos_features, neg_features, pos_neg_features = {'all': ['neu']}, {}, {}, {}
    neutral_features = {'all': ['neu']}
    for dim in test_dimensions:
        dimension_features[str(dim)] = modes
        pos_features[str(dim)] = pos_modes
        neg_features[str(dim)] = neg_modes
        pos_neg_features[str(dim)] = modes
    features = [dimension_features] #, pos_features, neg_features, pos_neg_features, neutral_features]

    combination_methods = ['comb_sum', 'feature_comb', 'comb_rank'] # 'comb_model', ['comb_sum', 'comb_rank', 'feature_comb']
    num_paragraphs = [[1, 3]]#, [1], [3]]
    algorithms = ['regression', 'ranking']#, 'ranking']#, 'mlp']

    unigrams = [False, True]#, True]#, True]
    kl_flags = ['kl']#, 'nokl', 'normkl']#[True, False]

    header = 'dim,unigrams_flag,combination_method,num_topic,optimal_num,paragraph_id,algorithm,modes,kl_flag,'
    header += 'model_type,error,kendall,pearson'
    output_file = open('report_main_table_{}.txt'.format(data_name), 'w+')
    output_file.write(header)

    for combination in combination_methods:
        for uni in unigrams:
            for topic_dims in topic_model_dims:
                for para in num_paragraphs:
                    for feature in features:
                        for algo in algorithms:
                            for kl in kl_flags:
                                output = single_experiment(test_dimensions, data_dir, uni, combination, topic_dims,
                                                           para, feature, algo, kl, topic_model_type)
                                output_file.write(output)
                                output_file.flush()


def main():
    run_experiments()


if __name__ == '__main__':
    main()
