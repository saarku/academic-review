from sklearn.neural_network import MLPRegressor
from feature_builder import FeatureBuilder
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
import joblib
from scipy.stats import kendalltau, pearsonr
import numpy as np
from svm_rank import SVMRank
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

'''
0. Try regression Tree
1. Check the results (print the model predictions (different aspects) and the actual predictions)
2. Prepare education data set and run experiments
3. Feature selection of the topic models
4. Score combination, Feature Combination, Feature Selection
'''


def single_experiment(test_dimensions, data_dir, unigrams_flag, combination_method, topic_model_dims, num_paragraphs,
                      dimension_features, algorithm):
    output_performance = ''
    builder = FeatureBuilder(data_dir)
    topics_dir, models_dir = data_dir + '/lda_vectors/', data_dir + '/models/'
    modes = set()
    for i in dimension_features: modes = modes.union(set(dimension_features[i]))
    modes = '_'.join([str(i) for i in modes])

    for dim in test_dimensions:
        model_dir = models_dir + 'dim.' + str(dim) + '.algo.' + algorithm
        all_features_train, all_features_test, feature_names, y_train, y_test = [], [], [], [], []

        if unigrams_flag:
            x_unigram_train, y_train, x_unigram_test, y_test = builder.build_unigram_features(dim)
            all_features_train.append(x_unigram_train)
            all_features_test.append(x_unigram_test)
            feature_names.append(str(0) + '_' + str(0) + '_' + str(0) + '_' + str(0) + '_true')

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
                        feature_names.append(str(topics)+'_'+str(para)+'_'+str(dim_feat)+'_'+str(mode)+'_false')

        if combination_method == 'feature_comb':
            train_features = sp.hstack(tuple(all_features_train), format='csr')
            test_features = sp.hstack(tuple(all_features_test), format='csr')
            model_dir += '.topics.' + '_'.join([str(i) for i in topic_model_dims])
            model_dir += '.para.' + '_'.join([str(i) for i in num_paragraphs])
            model_dir += '.dims.' + '_'.join([str(i) for i in dimension_features])
            model_dir += '.mode.' + '_'.join([str(i) for i in modes])
            model_dir += '.uni.' + str(unigrams_flag).lower()

            transformer = MinMaxScaler()
            transformer.fit(train_features.todense())
            train_features = transformer.transform(train_features.todense())
            test_features = transformer.transform(test_features.todense())

            if algorithm == 'regression':
                #clf = MLPRegressor(solver='sgd', max_iter=500, verbose=False).fit(train_features, y_train)
                clf = DecisionTreeRegressor()
                joblib.dump(clf, model_dir)
            else:
                clf = SVMRank()
                clf.fit(train_features, y_train, model_dir, 0.01)

            grades = clf.predict(test_features)

        else:
            grades = np.zeros((all_features_test[0].shape[0], 1), dtype=float)
            all_test_grades = []
            all_train_grades = []
            counter = 0
            for i in range(len(all_features_train)):
                args = feature_names[i].split('_')
                single_model_dir = model_dir
                single_model_dir += '.topics.' + args[0] + '.para.' + args[1] + '.dims.' + args[2] + '.mode.' + args[3]
                single_model_dir += '.uni.' + args[4]
                counter += 1

                train_features = all_features_train[i]
                test_features = all_features_test[i]
                transformer = MinMaxScaler()
                transformer.fit(train_features.todense())
                train_features = transformer.transform(train_features.todense())
                test_features = transformer.transform(test_features.todense())

                if algorithm == 'regression':
                    #clf = MLPRegressor(solver='sgd', max_iter=500, verbose=False).fit(train_features, y_train)
                    clf = DecisionTreeRegressor()
                    joblib.dump(clf, single_model_dir)
                else:
                    clf = SVMRank()
                    clf.fit(train_features, y_train, single_model_dir, 0.01)

                aspect_grades = clf.predict(test_features)
                aspect_grades = np.reshape(aspect_grades, (-1, 1))

                aspect_train_grades = clf.predict(train_features)
                aspect_train_grades = np.reshape(aspect_train_grades, (-1, 1))

                if combination_method == 'rank_comb':
                    aspect_grades = FeatureBuilder.grades_to_ranks(aspect_grades)

                grades += aspect_grades
                all_test_grades.append(aspect_grades)
                all_train_grades.append(aspect_train_grades)

            if combination_method == 'model_comb_linear':
                all_train_grades = np.hstack(all_train_grades)
                all_test_grades = np.hstack(all_test_grades)
                lr = LinearRegression().fit(all_train_grades, y_train)
                grades = lr.predict(all_test_grades)

            elif combination_method == 'model_comb_non_linear':
                all_train_grades = np.hstack(all_train_grades)
                all_test_grades = np.hstack(all_test_grades)
                t = MinMaxScaler()
                t.fit(all_train_grades)
                all_train_grades = t.transform(all_train_grades)
                all_test_grades = t.transform(all_test_grades)
                c = DecisionTreeRegressor()
                lr = c.fit(all_train_grades, y_train)
                #lr = MLPRegressor(solver='sgd', max_iter=500, verbose=False).fit(all_train_grades, y_train)
                grades = lr.predict(all_test_grades)
            else:
                grades /= float(counter)

        error = sqrt(mean_squared_error(y_test, grades))
        kendall, _ = kendalltau(y_test, grades)
        pearson, _ = pearsonr(y_test, np.reshape(grades, (1, -1)).tolist()[0])
        output_performance += '{},{},{},{},{},{},{},{},{},{},{}\n'.format(dim, unigrams_flag, combination_method,
                                                                          '_'.join([str(i) for i in topic_model_dims]),
                                                                          '_'.join([str(i) for i in num_paragraphs]),
                                                                          '_'.join([str(i) for i in dimension_features]),
                                                                          algorithm, modes, error, kendall, pearson)
    return output_performance


def main():
    data_dir = '/home/skuzi2/iclr17_dataset'
    test_dimensions = [1, 2, 3, 5, 6]
    topic_model_dims = [[10], [5]]
    modes, pos_modes, neg_modes = ['pos', 'neg'], ['pos'], ['neg']

    dimension_features = {'1': modes, '2': modes, '3': modes, '5': modes, '6': modes, 'all': ['neu']}
    pos_features = {'1': pos_modes, '2': pos_modes, '3': pos_modes, '5': pos_modes, '6': pos_modes}
    neg_features = {'1': neg_modes, '2': neg_modes, '3': neg_modes, '5': neg_modes, '6': neg_modes}
    pos_neg_features = {'1': modes, '2': modes, '3': modes, '5': modes, '6': modes}
    neutral_features = {'all': ['neu']}
    features = [pos_neg_features, neutral_features, dimension_features]

    combination_methods = ['model_comb_non_linear', 'feature_comb']
    num_paragraphs = [[1], [3], [1, 3]]
    algorithms = ['regression']
    unigrams = [False]
    header = 'test_dimension,unigrams,combination_method,num_topic_models,num_paragraphs'
    header += ',dimension_features,algorithm,modes,rmse,kendall,pearson\n'
    output_file = open('report_dt.txt', 'w+')
    output_file.write(header)

    for combination in combination_methods:
        for uni in unigrams:
            for topic_dims in topic_model_dims:
                for para in num_paragraphs:
                    for feature in features:
                        for algo in algorithms:
                            output = single_experiment(test_dimensions, data_dir, uni, combination, topic_dims, para,
                                                       feature, algo)
                            output_file.write(output)
                            output_file.flush()


if __name__ == '__main__':
    main()
