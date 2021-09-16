from feature_builder import FeatureBuilder
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from cross_validation import learn_model
from scipy.stats import kendalltau, pearsonr


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

grades = {}

for test_dim in test_dimensions:

    x_unigram_train, y_train, x_unigram_test, y_test, _, idx = builder.build_unigram_features(test_dim)
    grades[test_dim] = (idx, y_train)


for test_dim in test_dimensions:
    for other_dim in test_dimensions:
        if test_dim == other_dim: continue
        grades_1, idx_1 = grades[test_dim][1], grades[test_dim][0]
        grades_2, idx_2 = grades[other_dim][1], grades[other_dim][0]
        arr_1, arr_2 = [], []

        for i, id_1 in enumerate(idx_1):
            try:
                j = idx_2.index(id_1)
                arr_2.append(grades_2[j])
                arr_1.append(grades_1[i])
            except:
                continue

        kendall, _ = kendalltau(arr_1, arr_2)
        print('{},{},{}'.format(test_dim, other_dim, kendall))
















