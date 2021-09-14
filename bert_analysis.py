from feature_builder import FeatureBuilder
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from cross_validation import learn_model


def get_quarters(scores):
    scores = [(scores[i], i) for i in range(len(scores))]
    scores = sorted(scores, reverse=True)
    q = len(scores) // 3
    quarter_mapping = {(0, q): 1, (q, 2*q): 2, (2*q, len(scores)): 3}

    quarter_labels ={}
    for i, label in enumerate(scores):
        for tup in quarter_mapping:
            if tup[1] > i >= tup[0]:
                quarter_labels[label[1]] = quarter_mapping[tup]
    return quarter_labels


def grades_errors(predicted, labels):
    label_quarters = get_quarters(labels)
    predicted_quarters = get_quarters(predicted)
    mistakes = []
    for i in predicted_quarters:
        if predicted_quarters[i] != label_quarters[i]:
            mistakes.append(i)
    return mistakes


def get_overlap(error_1, error_2):
    error_1 = set(error_1)
    error_2 = set(error_2)
    return len(error_1.intersection(error_2))/len(error_1.union(error_2))


data_name = 'education'  # sys.argv[1]

bert_dir = {'iclr17': "dim.{}.algo.regression.uni.false.comb.feature_comb.model.bert.samedim.True.seed.1.samples.350.predict",
            'education': "dim.{}.algo.regression.uni.false.comb.feature_comb.model.bert.samedim.True.predict"}[data_name]

topics_dir = "dim.{}.algo.regression.uni.false.comb.comb_sum.model.lda.para.1_3.topics.cv.kl.kl.mode.neg_neu_pos.type.ovb.samedim.False.predict"


data_dir = '/home/skuzi2/{}_dataset'.format(data_name)
test_dimensions = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]

builder = FeatureBuilder(data_dir)

for test_dim in test_dimensions:

    x_unigram_train, y_train, x_unigram_test, y_test, _ = builder.build_unigram_features(test_dim)
    train_features = sp.hstack(tuple([x_unigram_train]), format='csr')
    test_features = sp.hstack(tuple([x_unigram_test]), format='csr')
    train_features, test_features = train_features.todense(), test_features.todense()
    transformer = MinMaxScaler()
    transformer.fit(train_features)
    train_features, test_features = transformer.transform(train_features), transformer.transform(test_features)
    clf = learn_model('regression', train_features, y_train, 'dim.{}.unigrams.model'.format(test_dim))
    unigram_grades = clf.predict(test_features)

    unigram_grades = unigram_grades.tolist()
    topic_grades = [float(i.rstrip('\n')) for i in open(data_dir + '/models/' + topics_dir.format(test_dim)).readlines()]
    bert_grades = [float(i.rstrip('\n')) for i in open(data_dir + '/models/' + bert_dir.format(test_dim)).readlines()]

    unigram_errors = grades_errors(unigram_grades, y_test)
    topic_errors = grades_errors(topic_grades, y_test)
    bert_errors = grades_errors(bert_grades, y_test)


    print('{},bert,{}'.format(test_dim, len(bert_errors)))
    print('{},topic,{}'.format(test_dim, len(topic_errors)))
    print('{},uni,{}'.format(test_dim, len(unigram_errors)))
    print('{},bert+uni,{}'.format(test_dim, get_overlap(bert_errors, unigram_errors)))
    print('{},topic+uni,{}'.format(test_dim, get_overlap(topic_errors, unigram_errors)))
    print('{},topic+bert,{}'.format(test_dim, get_overlap(bert_errors, topic_errors)))
















