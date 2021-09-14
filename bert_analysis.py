from feature_builder import FeatureBuilder
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from cross_validation import learn_model

data_name = 'iclr17'  # sys.argv[1]

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

    print('****************{}****************'.format(test_dim))
    print(y_test)
    print(unigram_grades)
    print(topic_grades)
    print(bert_grades)












