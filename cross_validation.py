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
from collections import defaultdict
from topic_models import TopicModels, get_vectors


def get_most_correlated_topics():
    data_name = sys.argv[1]
    data_dir = '/home/skuzi2/{}_dataset'.format(data_name)
    test_dims = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]
    output_file = open('correlations_{}.txt'.format(data_name), 'w')

    modes, dim_features = ['pos', 'neg'], {'all': ['neu']}
    for dim in test_dims: dim_features[str(dim)] = modes
    _, x, _, _, y, names = get_topic_model_vectors('cv', [1, 3], dim_features, 'ovb', 'kl', test_dims, data_dir,
                                                   same_dim_flag=False)

    for dim in x:
        for num in x[dim]:
            correlations = {}
            topics = x[dim][num]
            topic_names = names[dim][num]
            for i, topic_model in enumerate(topics):
                for topic_num in range(topic_model.shape[1]):
                    topic_name = topic_names[i] + '_' + str(topic_num)
                    f = topic_model[:,topic_num]
                    kendall, _ = kendalltau(f.todense(), y[dim])
                    correlations[topic_name] = kendall
            sorted_kendall = sorted(correlations, key=correlations.get, reverse=True)
            output_line = '{},{}'.format(dim, num)
            for i in sorted_kendall:
                output_line += ',' + i + ',' + str(correlations[i])
            output_file.write(output_line + '\n')
            output_file.flush()


def get_unigram_representations():
    test_dim = int(sys.argv[1])
    data_name = 'iclr17'
    data_dir = '/home/skuzi2/{}_dataset/'.format(data_name)
    builder = FeatureBuilder(data_dir)
    x_unigram_train, y_train, x_unigram_test, y_test, feature_names = builder.build_unigram_features(test_dim)

    correlations = {}
    for feature_id in range(x_unigram_test.shape[1]):
        features = x_unigram_test[:, feature_id].todense()
        kendall, _ = kendalltau(features, y_test)
        if kendall is not np.nan:
            correlations[feature_id] = kendall
    sorted_kendall = sorted(correlations, key=correlations.get, reverse=True)
    output_lines = ''

    for i in sorted_kendall[:100]:
        output_lines += feature_names[i] + ','
    output_lines += '\n'

    for i in sorted_kendall[len(sorted_kendall)-100:]:
        output_lines += feature_names[i] + ','
    print(output_lines)


def learn_model(algorithm, features, labels, model_dir):
    if algorithm == 'regression':
        clf = LinearRegression()
        clf.fit(features, labels)
        joblib.dump(clf, model_dir)

    elif algorithm == 'mlp_star':
        clf = MLPRegressor(batch_size=16, max_iter=500)
        clf.fit(features, labels)
        joblib.dump(clf, model_dir)

    else:
        clf = SVMRank()
        clf.fit(features, labels, model_dir, 0.01)
    return clf


def run_sum_comb_method(all_train_features, train_labels, all_test_features, algorithm, method):
    grades = np.zeros((all_test_features[0].shape[0], 1), dtype=float)
    train_grades = np.zeros((len(train_labels), 1), dtype=float)
    all_aspects_train, all_aspects_test = [], []
    all_coefficients = []
    all_features = []

    for i in range(len(all_train_features)):
        train_features, test_features = all_train_features[i], all_test_features[i]
        train_features, test_features = train_features.todense(), test_features.todense()

        transformer = MinMaxScaler()
        transformer.fit(train_features)
        train_features, test_features = transformer.transform(train_features), transformer.transform(test_features)
        all_features.append(test_features[1,:])
        all_features.append(test_features[2,:])

        temp_model_dir = 'val.' + str(time.time())
        clf = learn_model(algorithm, train_features, train_labels, temp_model_dir)
        coefficients = clf.coef_
        all_coefficients.append(coefficients)
        aspect_grades = clf.predict(test_features)
        aspect_grades = np.reshape(aspect_grades, (-1, 1))
        aspect_grades_train = clf.predict(train_features)
        aspect_grades_train = np.reshape(aspect_grades_train, (-1, 1))
        os.system('rm -rf ' + temp_model_dir)

        if method == 'comb_rank':
            aspect_grades = FeatureBuilder.grades_to_ranks(aspect_grades)
            aspect_grades_train = FeatureBuilder.grades_to_ranks(aspect_grades_train)

        if algorithm == 'ranking' and method == 'comb_sum':
            grades += softmax(aspect_grades)
            train_grades += softmax(aspect_grades_train)

        else:
            grades += aspect_grades
            train_grades += aspect_grades_train
            all_aspects_train.append(aspect_grades_train)
            all_aspects_test.append(aspect_grades)

    if method == 'comb_model':
        temp_model_dir = 'val.' + str(time.time())
        train_features = np.hstack(all_aspects_train)
        test_features = np.hstack(all_aspects_test)
        a_transformer = MinMaxScaler()
        a_transformer.fit(train_features)
        train_features = a_transformer.transform(train_features)
        test_features = a_transformer.transform(test_features)
        clf = learn_model('regression', train_features, train_labels, temp_model_dir)
        os.system('rm -rf ' + temp_model_dir)
        grades = clf.predict(test_features)
        train_grades = clf.predict(train_features)
        grades = np.reshape(grades, (-1, 1))
        train_grades = np.reshape(train_grades, (-1, 1))
        return grades, train_grades
    else:
        grades /= float(len(all_train_features))
        train_grades /= float(len(all_train_features))
        return grades, train_grades, all_coefficients, all_features


def cv_experiment(test_dimensions, data_dir, unigrams_flag, combination_method, train_vectors, test_vectors,
                  algorithm, model_name, cv_flag, eval_flag=True):

    builder = FeatureBuilder(data_dir) if unigrams_flag else None
    models_dir, args = data_dir + '/models/', model_name.split('.')
    header = ','.join([str(args[i]) for i in range(0, len(args), 2)])
    config = ','.join([str(args[i+1]) for i in range(0, len(args), 2)])
    header += ',dim,unigrams_flag,algo,combination_method,cv_flag,optimal_dim,error,pearson,kendall\n'
    output_performance = ''

    for test_dim in test_dimensions:
        y_train, y_test = FeatureBuilder.get_labels(data_dir, test_dim)
        model_dir = models_dir + 'dim.' + str(test_dim) + '.algo.' + algorithm + '.uni.' + str(unigrams_flag).lower()
        model_dir += '.comb.' + combination_method + '.' + model_name
        uni_features_train, uni_features_test = [], []

        if unigrams_flag:
            x_unigram_train, _, x_unigram_test, _, _ = builder.build_unigram_features(test_dim)
            uni_features_train.append(x_unigram_train)
            uni_features_test.append(x_unigram_test)

        if combination_method == 'feature_comb':
            optimal_dim, optimal_kendall = 0, -1
            for vec_dim in train_vectors[test_dim]:
                train_features = tuple(train_vectors[test_dim][vec_dim] + uni_features_train)
                train_features = sp.hstack(train_features, format='csr')
                all_train_ids = list(range(len(y_train)))
                random.seed(2)
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
                    optimal_dim = vec_dim

            train_features = sp.hstack(tuple(train_vectors[test_dim][optimal_dim] + uni_features_train), format='csr')
            test_features = sp.hstack(tuple(test_vectors[test_dim][optimal_dim] + uni_features_test), format='csr')
            train_features, test_features = train_features.todense(), test_features.todense()
            transformer = MinMaxScaler()
            transformer.fit(train_features)
            train_features, test_features = transformer.transform(train_features), transformer.transform(test_features)
            clf = learn_model(algorithm, train_features, y_train, model_dir + '.model')
            grades = clf.predict(test_features)
            grades = np.reshape(grades, (-1, 1))
            predication_dir = model_dir + '.predict'
            if not eval_flag: predication_dir += '.iclr'
            open(predication_dir, 'w').write('\n'.join([str(grades[i,0]) for i in range(grades.shape[0])]))

        else:
            optimal_dim, optimal_kendall = 0, -1

            for vec_dim in train_vectors[test_dim]:
                train_features = train_vectors[test_dim][vec_dim] + uni_features_train
                all_train_ids = list(range(len(y_train)))
                random.seed(2)
                random.shuffle(all_train_ids)
                val_split = int(len(all_train_ids) * 0.15)
                validation_ids, small_train_ids = all_train_ids[:val_split], all_train_ids[val_split:]
                validation_labels = [y_train[i] for i in range(len(y_train)) if i in validation_ids]
                small_train_labels = [y_train[i] for i in range(len(y_train)) if i in small_train_ids]

                small_train_features, validation_features = [], []
                for features in train_features:
                    small_train_features.append(features[small_train_ids, :])
                    validation_features.append(features[validation_ids, :])
                val_grades, _, _, _ = run_sum_comb_method(small_train_features, small_train_labels, validation_features,
                                                    algorithm, combination_method)
                kendall, _ = kendalltau(validation_labels, np.reshape(val_grades, (-1, 1)))
                if kendall > optimal_kendall:
                    optimal_kendall = kendall
                    optimal_dim = vec_dim

            train_features = train_vectors[test_dim][optimal_dim] + uni_features_train
            test_features = test_vectors[test_dim][optimal_dim] + uni_features_test
            grades, _, coefficients, all_features = run_sum_comb_method(train_features, y_train, test_features, algorithm, combination_method)
            #print(coefficients)
            print('----------')
            print(all_features)

            predication_dir = model_dir + '.predict'
            if not eval_flag: predication_dir += '.iclr'
            open(predication_dir, 'w').write('\n'.join([str(grades[i, 0]) for i in range(grades.shape[0])]))

        if eval_flag:
            error = sqrt(mean_squared_error(y_test, grades))
            kendall, _ = kendalltau(y_test, grades)
            pearson, _ = pearsonr(y_test, np.reshape(grades, (1, -1)).tolist()[0])

            performance = '{},{},{},{},{},{},{},{},{}\n'.format(test_dim, unigrams_flag, algorithm, combination_method,
                                                                cv_flag, optimal_dim, error, pearson, kendall)
            output_performance += config + ',' + performance

    return output_performance, header


def get_topic_model_vectors(num_topics, num_paragraphs, dimension_features, model_type, kl_flag, test_dims, data_dir,
                            same_dim_flag=True, train_flag=True):
    topics_dir = data_dir + '/lda_vectors_{}/'.format(model_type)
    train_features, test_features, model_names = {}, {}, {}
    y_train_dict, y_test_dict = {}, {}
    topic_model_dims = [5, 15, 25] if num_topics == 'cv' else [num_topics]
    builder = FeatureBuilder(data_dir) if train_flag else None

    for dim in test_dims:
        train_features[dim], test_features[dim] = defaultdict(list), defaultdict(list)
        model_names[dim] = defaultdict(list)
        for topics in topic_model_dims:
            for para in num_paragraphs:
                for dim_feat in dimension_features:
                    if str(dim_feat) != str(dim) and dim_feat != 'all' and same_dim_flag: continue
                    for mode in dimension_features[dim_feat]:
                        vec_dir = topics_dir
                        vec_dir += '{}_topics/dim.{}.mod.{}.para.{}.num.{}'.format(topics, dim_feat, mode, para, topics)
                        if kl_flag == 'kl' or kl_flag == 'normkl': vec_dir += '.kl'
                        norm = True if kl_flag == 'normkl' else False
                        x_topics_train, x_topics_test, y_train, y_test = [], [], [], []
                        if train_flag:
                            output = builder.build_topic_features(dim, vec_dir + '.train', vec_dir + '.test.val', para,
                                                                  norm=norm)
                            x_topics_train, y_train, x_topics_test, y_test = output[0], output[1], output[2], output[3]
                        else:
                            x_topics_test = get_vectors(vec_dir + '.test.val', para, norm)

                        y_train_dict[dim], y_test_dict[dim] = y_train, y_test
                        train_features[dim][topics].append(x_topics_train)
                        test_features[dim][topics].append(x_topics_test)
                        model_names[dim][topics].append('{}_{}_{}'.format(para, dim_feat, mode))

    modes = set()
    for i in dimension_features: modes = modes.union(set(dimension_features[i]))
    modes = '_'.join(sorted([str(i) for i in modes]))
    paragraph_id = '_'.join([str(i) for i in num_paragraphs])
    model_name = 'model.lda.para.{}.topics.{}.kl.{}.mode.{}.type.{}.samedim.{}'.format(paragraph_id, num_topics,
                                                                                       kl_flag, modes, model_type,
                                                                                       same_dim_flag)
    return train_features, test_features, model_name, y_train_dict, y_test_dict, model_names


def get_embedding_vectors(data_dir, arch, test_dims, vec_dim, same_dim_flag=True):
    vectors_dir = data_dir + '/embeddings_vectors/'
    config = 'wdim.20.epoch.5.batch.16.opt.adam.vocab.1000.length.100'
    train_features, test_features = {}, {}
    vec_dims = [5, 15, 25] if vec_dim == 'cv' else [vec_dim]
    builder = FeatureBuilder(data_dir)

    for test_dim in test_dims:
        train_features[test_dim], test_features[test_dim] = defaultdict(list), defaultdict(list)
        for dim_feat in test_dims:
            if dim_feat != test_dim and same_dim_flag: continue
            for d in vec_dims:
                curr_vectors_dir = vectors_dir + arch + '.dim.' + str(dim_feat) + '.ldim.' + str(d) + '.' + config
                output = builder.build_topic_features(test_dim, curr_vectors_dir + '.train', curr_vectors_dir + '.test.val', 1)
                x_train, _, x_test, _ = output[0], output[1], output[2], output[3]
                train_features[test_dim][d].append(x_train)
                test_features[test_dim][d].append(x_test)

    model_name = 'model.{}.dim.{}.samedim.{}'.format(arch, vec_dim, same_dim_flag)
    return train_features, test_features, model_name


def get_bert_vectors(data_dir, test_dims, num_samples, seed=0, same_dim_flag=True):
    if seed == 0:
        vectors_dir = data_dir + '/bert_embeddings/'
    else:
        vectors_dir = data_dir + '/bert_embeddings_{}/'.format(seed)

    train_features, test_features = {}, {}
    builder = FeatureBuilder(data_dir)

    for test_dim in test_dims:
        train_features[test_dim], test_features[test_dim] = defaultdict(list), defaultdict(list)
        for feature_dim in test_dims:
            if test_dim != feature_dim and same_dim_flag: continue
            bert_dir = vectors_dir + 'dim.' + str(feature_dim) + '.samples.' + str(num_samples)
            output = builder.build_topic_features(test_dim, bert_dir + '.train', bert_dir + '.test.val', 1)
            x_train, y_train, x_test, y_test = output[0], output[1], output[2], output[3]
            train_features[test_dim][25].append(x_train)
            test_features[test_dim][25].append(x_test)

    model_name = 'model.bert.samedim.{}.seed.{}.samples.{}'.format(same_dim_flag, seed, num_samples)
    return train_features, test_features, model_name


def run_topics_experiment():
    data_name = 'iclr17' # sys.argv[1]
    model_type = 'ovb'
    data_dir = '/home/skuzi2/{}_dataset'.format(data_name)
    test_dimensions = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]
    topic_model_dims = ['cv']
    same_dim_flag = [False]

    modes, pos_modes, neg_modes = ['pos', 'neg'], ['pos'], ['neg']
    dimension_features, pos_features, neg_features, pos_neg_features, pos_neu_features = {'all': ['neu']}, {}, {}, {}, {'all': ['neu']}
    neutral_features = {'all': ['neu']}
    for dim in test_dimensions:
        dimension_features[str(dim)] = modes
        pos_features[str(dim)] = pos_modes
        pos_neu_features[str(dim)] = pos_modes
        neg_features[str(dim)] = neg_modes
        pos_neg_features[str(dim)] = modes
    features = [dimension_features] #[pos_features, neg_features, pos_neg_features, neutral_features] #dimension_features] # dimension_features] #

    combination_methods = ['comb_sum'] #['feature_comb']#, 'comb_sum', 'comb_model'] # 'comb_model', ['comb_sum', 'comb_rank', 'feature_comb']
    num_paragraphs = [[1, 3]] #, [1], [3]] [1, 3]
    algorithms = ['regression']#, 'regression', 'ranking']#, 'ranking']#, 'ranking']#, 'mlp']

    unigrams = [False] #[True, False]#, True]#, True]
    kl_flags = ['kl'] #[True, False]

    output_file = open('report_{}.txt'.format(data_name), 'w+')
    output_lines, header = '', ''

    for topic_dim in topic_model_dims:
        for para in num_paragraphs:
            for feature in features:
                for kl in kl_flags:
                    for f in same_dim_flag:
                        args = get_topic_model_vectors(topic_dim, para, feature, model_type, kl, test_dimensions,
                                                       data_dir, same_dim_flag=f)
                        train_features, test_features, model_name = args[0], args[1], args[2]
                        feature_names = args[5]
                        #print(feature_names)

                        for combination in combination_methods:
                            for uni in unigrams:
                                for algo in algorithms:
                                    output, header = cv_experiment(test_dimensions, data_dir, uni, combination,
                                                                   train_features, test_features, algo, model_name,
                                                                   topic_dim)
                                    output_lines += output
                                    print(output)
    output_file.write(header)
    output_file.write(output_lines)


def run_embeddings_experiment():
    data_name = 'iclr17' #sys.argv[1]
    arch = 'bert_250' #sys.argv[2]
    num_samples = arch.split('_')[1]

    same_dim_flag = [True]
    data_dir = '/home/skuzi2/{}_dataset'.format(data_name)
    test_dimensions = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1]}[data_name] #[1, 2, 3, 5, 6]
    combination_methods = ['feature_comb'] #'comb_sum',
    algorithms = ['regression']
    unigrams = [False] #, True
    output_file = open('report_{}_{}_{}.txt'.format(arch, data_name, num_samples), 'w+')
    output_lines, header = '', ''

    for f in same_dim_flag:
        if arch in ['lstm', 'cnn']:
            train_features, test_features, model_name = get_embedding_vectors(data_dir, arch, test_dimensions, 'cv',
                                                                              same_dim_flag=f)
        else:
            train_features, test_features, model_name = get_bert_vectors(data_dir, test_dimensions, num_samples,
                                                                         same_dim_flag=f)

        for uni in unigrams:
            for combination in combination_methods:
                for algo in algorithms:
                    output, header = cv_experiment(test_dimensions, data_dir, uni, combination, train_features,
                                                   test_features, algo, model_name, 'cv')
                    output_lines += output
                    print(output)

    output_file.write(header)
    output_file.write(output_lines)


def run_bert_diagnose_experiment():
    data_name = 'iclr17' #sys.argv[1]
    data_dir = '/home/skuzi2/{}_dataset'.format(data_name)
    test_dimensions = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [3]}[data_name] #[1, 2, 3, 5, 6]
    output_file = open('report_bert_diagnose_{}.txt'.format(data_name), 'w+')
    output_lines, header = '', ''

    for seed in [0]:
        for num_samples in [300, 350]:
            train_features, test_features, model_name = get_bert_vectors(data_dir, test_dimensions, num_samples,
                                                                         seed=seed)
            output, header = cv_experiment(test_dimensions, data_dir, False, 'feature_comb', train_features,
                                           test_features, 'regression', model_name, 'cv')
            output_lines += output
            print(output)

    output_file.write(header)
    output_file.write(output_lines)


def neural_comb():
    data_name = sys.argv[1]

    dims = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]
    combination_methods = ['comb_sum', 'feature_comb']
    unigrams = [False, True]
    data_dir = '/home/skuzi2/{}_dataset'.format(data_name)
    feature_dims, algorithm = [5, 15, 25], 'regression'
    output_file = open('report_fusion_{}.txt'.format(data_name), 'w+')
    modes, dimension_features = ['pos', 'neg'], {'all': ['neu']}
    for dim in dims: dimension_features[str(dim)] = modes
    output_lines, header = '', ''

    multi_dict = {'iclr17': {'lstm': {1:'multi', 2:'multi', 3:'multi', 5:'single', 6:'single'},
                             'bert': {1:'single', 2:'single', 3:'multi', 5:'multi', 6:'multi'},
                             'lda': {1:'multi', 2:'multi', 3:'single', 5:'multi', 6:'single'}},
                  'education': {'lstm': {0: 'multi', 1: 'multi', 2: 'multi', 3: 'multi', 4: 'single', 5: 'single', 6: 'single'},
                                'bert': {0: 'multi', 1: 'multi', 2: 'multi', 3: 'multi', 4: 'multi', 5: 'multi', 6: 'multi'},
                                'lda': {0: 'multi', 1: 'multi', 2: 'single', 3: 'multi', 4: 'single', 5: 'multi', 6: 'single'}}
                  }

    print('load lstm')
    lstm_train_single, lstm_test_single, _ = get_embedding_vectors(data_dir, 'lstm', dims, 'cv', same_dim_flag=True)
    lstm_train_multi, lstm_test_multi, _ = get_embedding_vectors(data_dir, 'lstm', dims, 'cv', same_dim_flag=False)
    lstm_train = {'multi': lstm_train_multi, 'single': lstm_train_single}
    lstm_test = {'multi': lstm_test_multi, 'single': lstm_test_single}

    print('load bert')
    bert_train_single, bert_test_single, _ = get_bert_vectors(data_dir, dims, same_dim_flag=True)
    bert_train_multi, bert_test_multi, _ = get_bert_vectors(data_dir, dims, same_dim_flag=False)
    bert_train = {'multi': bert_train_multi, 'single': bert_train_single}
    bert_test = {'multi': bert_test_multi, 'single': bert_test_single}

    print('load lda')
    args = get_topic_model_vectors('cv', [1, 3], dimension_features, 'ovb', 'kl', dims, data_dir, same_dim_flag=True)
    lda_train_single, lda_test_single = args[0], args[1]
    args = get_topic_model_vectors('cv', [1, 3], dimension_features, 'ovb', 'kl', dims, data_dir, same_dim_flag=False)
    lda_train_multi, lda_test_multi = args[0], args[1]
    lda_train = {'multi': lda_train_multi, 'single': lda_train_single}
    lda_test = {'multi': lda_test_multi, 'single': lda_test_single}

    print('combine features')
    train_features, test_features = {}, {}
    for lda_dim in feature_dims:
        for lstm_dim in feature_dims:
            comb = '{}_{}'.format(lda_dim, lstm_dim)
            for test_dim in lda_train_multi:

                if test_dim not in train_features:
                    train_features[test_dim], test_features[test_dim] = defaultdict(list), defaultdict(list)

                multi_lda = multi_dict[data_name]['lda'][test_dim]
                train_features[test_dim][comb] += lda_train[multi_lda][test_dim][lda_dim]
                test_features[test_dim][comb] += lda_test[multi_lda][test_dim][lda_dim]

                multi_bert = multi_dict[data_name]['bert'][test_dim]
                train_features[test_dim][comb] += bert_train[multi_bert][test_dim][25]
                test_features[test_dim][comb] += bert_test[multi_bert][test_dim][25]

                multi_lstm = multi_dict[data_name]['lstm'][test_dim]
                train_features[test_dim][comb] += lstm_train[multi_lstm][test_dim][lstm_dim]
                test_features[test_dim][comb] += lstm_test[multi_lstm][test_dim][lstm_dim]

    for uni in unigrams:
        for comb in combination_methods:
            print('{}_{}'.format(uni, comb))
            output, header = cv_experiment(dims, data_dir, uni, comb, train_features, test_features, algorithm,
                                           'fusion.true', 'cv')
            output_lines += output

    output_file.write(header)
    output_file.write(output_lines)


def get_acl_scores():
    model_type = 'ovb'
    train_data_dir = '/home/skuzi2/iclr17_dataset'
    test_data_dir = '/home/skuzi2/iclrlarge_dataset'
    test_dimensions = [1, 2, 3, 5, 6]

    same_dim_flag = [True, False]

    modes, dimension_features = ['pos', 'neg'], {'all': ['neu']}
    for dim in test_dimensions: dimension_features[str(dim)] = modes
    combination_methods = ['comb_sum', 'feature_comb']
    para = [1, 3]

    for f in same_dim_flag:
        print('loading training features ({})'.format(f))
        args = get_topic_model_vectors('cv', para, dimension_features, model_type, 'kl', test_dimensions,
                                       train_data_dir, same_dim_flag=f, train_flag=True)
        train_features, model_name = args[0], args[2]

        print('loading test features ({})'.format(f))
        args = get_topic_model_vectors('cv', para, dimension_features, model_type, 'kl', test_dimensions,
                                       test_data_dir, same_dim_flag=f, train_flag=False)
        test_features = args[1]

        for combination in combination_methods:
            print('learning: {},{}'.format(f, combination))
            cv_experiment(test_dimensions, train_data_dir, False, combination, train_features, test_features,
                          'regression', model_name, 'cv', eval_flag=False)


def main():
    run_bert_diagnose_experiment()


if __name__ == '__main__':
    main()
