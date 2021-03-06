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

'''
1. Run for num of topics in {5, 15, 25}
2. Analyse results - sensitivity to num paragraphs?
2. Tune number of paragraphs (in {1, 3, 1_3} and num topics (3 options) on a validation set - for pearson and kendall separately.
3. maybe focus on results without kl.
Interesting further study: feature combination approaches, feature selection.
'''


def single_experiment(test_dimensions, data_dir, unigrams_flag, combination_method, topic_model_dims, num_paragraphs,
                      dimension_features, algorithm, kl_flag, model_type):
    output_performance = ''
    builder = FeatureBuilder(data_dir)
    topics_dir, models_dir = data_dir + '/lda_vectors_{}/'.format(model_type), data_dir + '/models/'
    modes = set()
    for i in dimension_features: modes = modes.union(set(dimension_features[i]))
    modes = '_'.join(sorted([str(i) for i in modes]))

    for dim in test_dimensions:
        #debug_file = open('debug' + str(dim) + '.txt', 'w')
        model_dir = models_dir + 'dim.' + str(dim) + '.algo.' + algorithm
        comb_model_dir = model_dir
        comb_model_dir += '.topics.' + '_'.join([str(i) for i in topic_model_dims])
        comb_model_dir += '.para.' + '_'.join([str(i) for i in num_paragraphs])
        comb_model_dir += '.mode.' + modes
        comb_model_dir += '.kl.' + str(kl_flag).lower()
        comb_model_dir += '.type.' + str(model_type).lower()
        comb_model_dir += '.uni.' + str(unigrams_flag).lower()
        all_features_train, all_features_test, feature_names, y_train, y_test = [], [], [], [], []

        if unigrams_flag:
            x_unigram_train, y_train, x_unigram_test, y_test, _ = builder.build_unigram_features(dim)
            all_features_train.append(x_unigram_train)
            print(x_unigram_train.shape)
            all_features_test.append(x_unigram_test)
            feature_names.append('0_0_0_0_0_0_true')

        for topics in topic_model_dims:
            for para in num_paragraphs:
                for dim_feat in dimension_features:
                    for mode in dimension_features[dim_feat]:
                        vec_dir = topics_dir
                        vec_dir += '{}_topics/dim.{}.mod.{}.para.{}.num.{}'.format(topics, dim_feat, mode, para, topics)
                        if kl_flag == 'kl' or kl_flag == 'norm_kl': vec_dir += '.kl'
                        norm = True if kl_flag == 'norm_kl' else False
                        output = builder.build_topic_features(dim, vec_dir + '.train', vec_dir + '.test.val', para,
                                                              norm=norm)
                        x_topics_train, y_train, x_topics_test, y_test = output[0], output[1], output[2], output[3]
                        all_features_train.append(x_topics_train)
                        all_features_test.append(x_topics_test)
                        feature_names.append('{}_{}_{}_{}_{}_{}_false'.format(topics, para, mode, kl_flag, model_type, dim_feat))

        if combination_method == 'feature_comb':
            comb_model_dir += '.comb.' + combination_method
            train_features = sp.hstack(tuple(all_features_train), format='csr')
            test_features = sp.hstack(tuple(all_features_test), format='csr')

            transformer = MinMaxScaler()
            train_features, test_features = train_features.todense(), test_features.todense()
            transformer.fit(train_features)
            train_features = transformer.transform(train_features)
            test_features = transformer.transform(test_features)

            if algorithm == 'regression':
                clf = LinearRegression()
                clf.fit(train_features, y_train)
                joblib.dump(clf, comb_model_dir)
            elif algorithm == 'mlp':
                clf = MLPRegressor(batch_size=16, max_iter=500)
                clf.fit(train_features, y_train)
                joblib.dump(clf, comb_model_dir)
            else:
                clf = SVMRank()
                clf.fit(train_features, y_train, comb_model_dir, 0.01)

            grades = clf.predict(test_features)
            grades = np.reshape(grades, (-1, 1))
            open(comb_model_dir + '.predictions', 'w').write('\n'.join([str(grades[i,0])
                                                                        for i in range(grades.shape[0])]))

        else:
            grades = np.zeros((all_features_test[0].shape[0], 1), dtype=float)
            all_test_grades, all_train_grades = [], []
            counter = 0
            for i in range(len(all_features_train)):
                args = feature_names[i].split('_')
                single_model_dir = model_dir
                single_model_dir += '.topics.' + args[0] + '.para.' + args[1] + '.mode.' + args[2]
                single_model_dir += '.kl.' + args[3].lower() + '.type.' + args[4] + '.uni.' + args[5] + '.dimfeat.' + \
                                    args[6] + '.comb.single'
                counter += 1

                train_features = all_features_train[i]
                test_features = all_features_test[i]
                train_features = train_features.todense()
                test_features = test_features.todense()

                transformer = MinMaxScaler()
                transformer.fit(train_features)
                train_features = transformer.transform(train_features)
                test_features = transformer.transform(test_features)

                if algorithm == 'regression':
                    clf = LinearRegression()
                    clf.fit(train_features, y_train)
                    joblib.dump(clf, single_model_dir)
                elif algorithm == 'mlp':
                    clf = MLPRegressor(solver='sgd', verbose=False)
                    clf.fit(train_features, y_train)
                    joblib.dump(clf, single_model_dir)
                else:
                    clf = SVMRank()
                    clf.fit(train_features, y_train, single_model_dir, 0.01)

                aspect_grades = clf.predict(test_features)
                aspect_grades = np.reshape(aspect_grades, (-1, 1))
                aspect_train_grades = clf.predict(train_features)
                aspect_train_grades = np.reshape(aspect_train_grades, (-1, 1))

                open(single_model_dir + '.predictions', 'w').write('\n'.join([str(aspect_grades[i, 0])
                                                                              for i in range(aspect_grades.shape[0])]))

                if combination_method == 'comb_rank':
                    aspect_grades = FeatureBuilder.grades_to_ranks(aspect_grades)

                if algorithm == 'ranking' and combination_method == 'comb_sum':
                    grades += softmax(aspect_grades)
                else:
                    grades += aspect_grades

                all_test_grades.append(aspect_grades)
                all_train_grades.append(aspect_train_grades)

            if combination_method == 'comb_model':
                l = LinearRegression()
                grades_train = np.hstack(all_train_grades)
                m = MinMaxScaler()
                m.fit(grades_train)
                grades_train = m.transform(grades_train)
                grades_test = m.transform(np.hstack(all_test_grades))
                l.fit(grades_train, y_train)
                grades = l.predict(grades_test)
                grades = np.reshape(grades, (-1, 1))
            else:
                grades /= float(counter)

            open(comb_model_dir + '.comb.' + combination_method + '.predictions', 'w').write(
                '\n'.join([str(grades[i, 0]) for i in range(grades.shape[0])]))

            '''
            header = ''
            for name in feature_names:
                header += name + ','
            header += 'final,golden\n'
            debug_file.write(header)
            all_test_grades = np.hstack(all_test_grades + [grades, np.reshape(np.asarray(y_test), (-1,1))])

            for i in range(all_test_grades.shape[0]):
                line = ''
                for j in range(all_test_grades.shape[1]):
                    line += str(all_test_grades[i, j]) + ','
                debug_file.write(line + '\n')
            '''

        error = sqrt(mean_squared_error(y_test, grades))
        kendall, _ = kendalltau(y_test, grades)
        pearson, _ = pearsonr(y_test, np.reshape(grades, (1, -1)).tolist()[0])
        performance = '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(dim, unigrams_flag, combination_method,
                                                                     '_'.join([str(i) for i in topic_model_dims]),
                                                                     '_'.join([str(i) for i in num_paragraphs]),
                                                                     algorithm, modes, kl_flag, model_type, error,
                                                                     kendall, pearson)
        output_performance += performance
    return output_performance


def lstm_experiment(test_dimensions, data_dir, algorithm, lstm_model_name):
    output_performance = ''
    builder = FeatureBuilder(data_dir)
    lstm_dir, models_dir = data_dir + '/embeddings_vectors/', data_dir + '/lstm_models/'

    for dim in test_dimensions:
        model_dir = models_dir + 'dim.' + str(dim) + '.algo.' + algorithm + '.' + lstm_model_name
        vectors_dir = lstm_dir + 'cnn.dim.' + str(dim) + '.' + lstm_model_name
        output = builder.build_topic_features(dim, vectors_dir + '.train', vectors_dir + '.test.val', 1)
        x_train, y_train, x_test, y_test = output[0], output[1], output[2], output[3]

        transformer = MinMaxScaler()
        train_features, test_features = x_train.todense(), x_test.todense()
        transformer.fit(train_features)
        train_features = transformer.transform(train_features)
        test_features = transformer.transform(test_features)

        if algorithm == 'regression':
            clf = LinearRegression()
            clf.fit(train_features, y_train)
            joblib.dump(clf, model_dir)
        else:
            clf = SVMRank()
            clf.fit(train_features, y_train, model_dir, 0.01)

        grades = clf.predict(test_features)
        grades = np.reshape(grades, (-1, 1))
        open(model_dir + '.predictions', 'w').write('\n'.join([str(grades[i,0]) for i in range(grades.shape[0])]))

        error = sqrt(mean_squared_error(y_test, grades))
        kendall, _ = kendalltau(y_test, grades)
        pearson, _ = pearsonr(y_test, np.reshape(grades, (1, -1)).tolist()[0])
        performance = '{},{},{},{},{},{}\n'.format(dim, algorithm, lstm_model_name, error, kendall, pearson)
        output_performance += performance
    return output_performance


def run_experiments():
    data_name = {1: 'iclr17', 2: 'education'}[int(sys.argv[1])]
    topic_model_type = sys.argv[2]
    data_dir = '/home/skuzi2/{}_dataset'.format(data_name)
    test_dimensions = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]
    topic_model_dims = [[int(sys.argv[3])]]

    modes, pos_modes, neg_modes = ['pos', 'neg'], ['pos'], ['neg']
    dimension_features, pos_features, neg_features, pos_neg_features = {'all': ['neu']}, {}, {}, {}
    neutral_features = {'all': ['neu']}
    for dim in test_dimensions:
        dimension_features[str(dim)] = modes
        pos_features[str(dim)] = pos_modes
        neg_features[str(dim)] = neg_modes
        pos_neg_features[str(dim)] = modes
    features = [dimension_features, pos_features, neg_features, pos_neg_features, neutral_features]

    combination_methods = ['comb_sum', 'feature_comb', 'comb_rank'] # 'comb_model', ['comb_sum', 'comb_rank', 'feature_comb']
    num_paragraphs = [[1, 3], [1], [3]]
    algorithms = ['regression', 'ranking']#, 'ranking']#, 'mlp']

    unigrams = [False, True]#, True]#, True]
    kl_flags = ['kl', 'no_kl', 'norm_kl']#[True, False]
    header = 'test_dimension,unigrams,combination_method,num_topic_models,num_paragraphs'
    header += ',algorithm,modes,kl,rmse,kendall,pearson\n'
    output_file = open('report_{}_{}_{}.txt'.format(data_name, topic_model_type, topic_model_dims[0][0]), 'w+')
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


def unigram_baseline():
    data_name = sys.argv[1] #{1: 'iclr17', 2: 'education'}[int(sys.argv[1])]
    data_dir = '/home/skuzi2/{}_dataset'.format(data_name)
    test_dimensions = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]
    algorithms = ['regression']
    header = 'test_dimension,unigrams,combination_method,num_topic_models,num_paragraphs'
    header += ',algorithm,modes,kl,rmse,kendall,pearson\n'
    output_file = open('report_unigrams_{}.txt'.format(data_name), 'w+')
    output_file.write(header)

    for algo in algorithms:
        output = single_experiment(test_dimensions, data_dir, True, 'feature_comb', [], [1], {'all': ['neu']}, algo,
                                   False, 'ovb')
        output_file.write(output)
        output_file.flush()


def lstm_baseline():
    data_name = sys.argv[1]
    dimensions = [5, 15, 25]
    w_dims = [20]
    epochs = [5]
    batch_sizes = [16]
    vocabs = [1000]
    lengths = [100]
    optimizers = ['adam']
    grade_dims = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]
    algorithms = ['regression', 'ranking']
    data_dir = '/home/skuzi2/{}_dataset'.format(data_name)

    header = 'dim,algorithm,lstm_model_name,error,kendall,pearson\n'
    output_file = open('report_cnn_{}.txt'.format(data_name), 'w+')
    output_file.write(header)

    for lstm_dim in dimensions:
        for word_dim in w_dims:
            for epoch in epochs:
                for batch in batch_sizes:
                    for opt in optimizers:
                        for algorithm in algorithms:
                            for v in vocabs:
                                for l in lengths:
                                    lstm_model_name = 'ldim.{}.wdim.{}.epoch.{}.batch.{}.opt.{}.vocab.{}.length.{}'.format(lstm_dim,
                                                                                                                 word_dim,
                                                                                                                 epoch,
                                                                                                                 batch,
                                                                                                                 opt, v, l)
                                    output = lstm_experiment(grade_dims, data_dir, algorithm, lstm_model_name)
                                    output_file.write(output)
                                    output_file.flush()


def main():
    unigram_baseline()


if __name__ == '__main__':
    main()
