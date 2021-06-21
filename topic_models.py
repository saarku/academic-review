from utils import to_sparse, from_sparse, pre_process_text
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import numpy as np
import joblib
import sys
from collections import defaultdict
from scipy.stats import kendalltau


def get_topic_representations():
    data_name = 'iclr17'
    num_topics = '25'
    topic_identifiers_2 = ['1_6_pos_23', '1_1_pos_12', '1_1_neg_7', '1_2_pos_13']
    topic_identifiers_1 = ['1_all_neu_0', '1_3_neg_4', '1_5_pos_17', '1_all_neu_15']

    topic_identifiers = topic_identifiers_1
    # 'para_dimfeat_mode_num'
    data_dir = '/home/skuzi2/{}_dataset/'.format(data_name)
    topic_words = {}

    for topic_id in topic_identifiers:
        args = topic_id.split('_')
        topic_data_dir = data_dir + 'data_splits/dim.{}.mod.{}.para.{}.train.text'.format(args[1], args[2], args[0])
        tm = TopicModels(topic_data_dir, topic_data_dir)
        topic_model_dir = data_dir + 'lda_models/{}_topics/dim.{}.mod.{}.para.{}.num.{}/model'.format(num_topics,
                                                                                                      args[1], args[2],
                                                                                                      args[0],
                                                                                                      num_topics)
        words = tm.generate_topic_words(topic_model_dir)
        topic_words[topic_id] = words[int(args[3])]

    final_words = defaultdict(list)
    for topic_id in topic_words:
        words_set = set(topic_words[topic_id])
        for other_topic_id in topic_words:
            if other_topic_id != topic_id: words_set = words_set - set(topic_words[other_topic_id])
        for word in topic_words[topic_id]:
            if word in words_set:
                final_words[topic_id].append(word)

    for topic_id in final_words:
        output_line = topic_id
        for word in final_words[topic_id][:20]:
            output_line += ',' + word
        print(output_line)


def get_topics_vec(dists_dir, labels, dimension_id, num_paragraphs, norm):
    """ Read topic distributions from a file and construct a sparse matrix.

    :param dists_dir: a file with the distributions.
    :param dimension_id: the dimension for grading.
    :param num_paragraphs: number of paragraphs.
    :param norm: (bool) whether to to l1 norm.
    :return: csr_matrix.
    """
    all_vectors = get_vectors(dists_dir, num_paragraphs, norm)
    modified_grades = []
    indexes = []

    for i in range(all_vectors.shape[0]):
        grade = labels[i, dimension_id]
        if grade > 0:
            indexes.append(i)
            modified_grades.append(grade)

    return all_vectors[indexes, :], modified_grades


def get_vectors(dists_dir, num_paragraphs, norm):
    """ Read topic distributions from a file and construct a sparse matrix.

    :param dists_dir: a file with the distributions.
    :param num_paragraphs: number of paragraphs.
    :param norm: (bool) whether to to l1 norm.
    :return: csr_matrix.
    """
    all_topic_vectors = []
    distributions = open(dists_dir, 'r').readlines()
    for i in range(0, len(distributions), num_paragraphs):
        single_vec = {}
        for j in range(num_paragraphs):
            if i+j >= len(distributions): continue
            args = distributions[i+j].rstrip('\n').rstrip(']').lstrip('[').split('),') #here
            for a in args:
                index = int(a.split(',')[0].split('(')[1])
                number = float(a.split(',')[1].rstrip(')'))
                single_vec[index] = max(single_vec.get(index, -float('inf')), number)
        if norm:
            normalizer = sum(single_vec.values())
            if normalizer <= 0: normalizer = 1
            for j in single_vec:
                single_vec[j] /= normalizer

        single_vec = list(single_vec.items())
        all_topic_vectors.append(single_vec)

    num_topics = len(all_topic_vectors[0])
    return to_sparse(all_topic_vectors, (len(all_topic_vectors), num_topics))


class TopicModels:

    def __init__(self, data_dir, vocabulary_data_dir):
        self.data_lines = []
        for i, line in enumerate(open(data_dir, 'r').read().split('\n')):
            self.data_lines.append(pre_process_text(line))

        if vocabulary_data_dir is not None:
            self.vocab_lines = [pre_process_text(line) for line in open(vocabulary_data_dir, 'r').read().split('\n')][0:-1]

    def upload_vocab(self, vocabulary_data_dir):
        self.vocab_lines = [pre_process_text(line) for line in open(vocabulary_data_dir, 'r').read().split('\n')][0:-1]

    def learn_lda(self, num_topics, output_dir, model_type):
        """ Learn an LDA topic model.

        :param num_topics: (int) number file with the distributions.
        :param output_dir: (string) directory for the output model.
        :param model_type: (string) either gibbs or ovb.
        :return: None.
        """
        count_vector = CountVectorizer()
        train_data = count_vector.fit_transform(self.data_lines)

        if model_type == 'gibbs':
            train_data = from_sparse(train_data)
            lda = LdaModel(train_data, num_topics=num_topics)
            lda.save(output_dir)
        elif model_type == 'ovb':
            lda = LatentDirichletAllocation(n_components=num_topics)
            lda.fit(train_data)
            joblib.dump(lda, output_dir + '.ovb')
        else:
            print(str(model_type) + ' not supported')
            return -1

    def generate_topic_dists(self, topic_model_dir, output_dir, model_type):
        """ Generate topic representation for training/test data.

        :param topic_model_dir: (string) directory of the topic model.
        :param output_dir: (string).
        :param model_type: (string) either gibbs or ovb.
        :return: None.
        """
        count_vector_lda = CountVectorizer()
        count_vector_lda.fit(self.vocab_lines)
        x_lda_counts = count_vector_lda.transform(self.data_lines)

        if model_type == 'gibbs':
            lda_model = LdaModel.load(topic_model_dir)
            x_vectors = from_sparse(x_lda_counts)
            lda_file = open(output_dir, 'w+')
            for lda_vector in x_vectors:
                lda_file.write(str(lda_model.get_document_topics(lda_vector, minimum_probability=0.0)) + '\n')
            lda_file.close()

        elif model_type == 'ovb':
            lda = joblib.load(topic_model_dir + '.ovb')
            topics = lda.transform(x_lda_counts)
            lda_file = open(output_dir, 'w+')
            for i in range(topics.shape[0]):
                line = '['
                for j in range(topics.shape[1]):
                    line += '({}, {}), '.format(j, topics[i, j])
                line = line.rstrip(', ') + ']\n'
                lda_file.write(line)
        else:
            print(str(model_type) + ' not supported')
            return -1

    def generate_topic_words(self, topic_model_dir):
        """ Generate topic representation for training/test data.

        :param topic_model_dir: (string) directory of the topic model.
        :param output_dir: (string).
        :param model_type: (string) either gibbs or ovb.
        :return: None.
        """
        n_top_words = 100
        count_vector_lda = CountVectorizer()
        count_vector_lda.fit(self.vocab_lines)
        feature_names = count_vector_lda.get_feature_names()

        lda = joblib.load(topic_model_dir + '.ovb')
        words = {}

        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            words[topic_idx] = top_features
        return words

    def generate_topic_kl(self, topic_model_dir, output_dir, model_type):
        """ Generate topic representation (with kl-divergence) for training/test data.

        :param topic_model_dir: (string) directory of the topic model.
        :param output_dir: (string).
        :param model_type: (string) either gibbs or ovb.
        :return: None.
        """
        lda_file = open(output_dir, 'w+')

        count_vector_lda = CountVectorizer()
        count_vector_lda.fit(self.vocab_lines)
        x_lda_counts = count_vector_lda.transform(self.data_lines)
        x_vectors = from_sparse(x_lda_counts)

        if model_type == 'gibbs':
            lda_model = LdaModel.load(topic_model_dir)
            all_topics = lda_model.get_topics()
        elif model_type == 'ovb':
            lda_model = joblib.load(topic_model_dir + '.ovb')
            all_topics = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
        else:
            print(str(model_type) + ' not supported')
            return -1

        for doc_vec in x_vectors:
            doc_vec = dict(doc_vec)
            normalizer = sum(doc_vec.values())
            for term in doc_vec: doc_vec[term] /= normalizer
            doc_topics = []

            for topic_id in range(all_topics.shape[0]):
                kl_score = 0
                for term in doc_vec:
                    if all_topics[topic_id, term] > 0:
                        kl_score += doc_vec[term] * np.log(doc_vec[term]/all_topics[topic_id, term])
                doc_topics.append((topic_id, kl_score))

            lda_file.write(str(doc_topics) + '\n')
            lda_file.flush()
        lda_file.close()


def main():

    dataset = sys.argv[1]
    topics = [5, 15, 25]
    modes = ['pos', 'neg']
    dimensions_ed = {'0': modes, '1': modes, '2': modes, '3': modes, '4': modes, '5': modes, '6': modes, 'all': ['neu']}
    dimensions_ic = {'1': modes, '2': modes, '3': modes, 'all': ['neu'], '5': modes, '6': modes}
    paragraphs = ['1', '3']
    base_dir = '../{}_dataset/'.format(dataset)
    dimensions = dimensions_ed if dataset == 'education' else dimensions_ic
    model_types = ['ovb']  # ovb or gibbs

    learn_flag = False
    infer_flag = False
    infer_test_flag = True

    if learn_flag:
        for dim in dimensions:
            for mode in dimensions[dim]:
                for para in paragraphs:
                    for model_type in model_types:
                        for topic in topics:
                            data_dir = base_dir + 'data_splits/dim.{}.mod.{}.para.{}.train.text'.format(dim, mode, para)
                            model_dir = base_dir + 'lda_models/{}_topics/'.format(topic)
                            model_dir += 'dim.{}.mod.{}.para.{}.num.{}/'.format(dim, mode, para, topic)
                            os.mkdir(model_dir) if not os.path.exists(model_dir) else None
                            print('learn ' + model_dir)
                            tm = TopicModels(data_dir, data_dir)
                            tm.learn_lda(topic, model_dir + '/model', model_type)

    if infer_flag:
        for para in paragraphs:
            for dim in dimensions:
                for mode in dimensions[dim]:
                    for model_type in model_types:
                        for topic in topics:
                            train_data_dir = base_dir + '/data_splits/dim.{}.mod.{}.para.{}.train.text'.format(dim, mode, para)
                            test_data_dir = base_dir + '/data_splits/dim.{}.mod.{}.para.{}.test.val.text'.format(dim, mode, para)

                            model_dir = base_dir + '/lda_models/'
                            model_dir += '{}_topics/dim.{}.mod.{}.para.{}.num.{}/model'.format(topic, dim, mode, para,topic)
                            vectors_dir = base_dir + '/lda_vectors_{}/'.format(model_type)
                            vectors_dir += '{}_topics/dim.{}.mod.{}.para.{}.num.{}'.format(topic, dim, mode, para, topic)
                            print('infer ' + vectors_dir)

                            tm = TopicModels(train_data_dir, train_data_dir)
                            tm.generate_topic_kl(model_dir, vectors_dir + '.kl.train', model_type)
                            tm.generate_topic_dists(model_dir, vectors_dir + '.train', model_type)

                            tm = TopicModels(test_data_dir, train_data_dir)
                            tm.generate_topic_kl(model_dir, vectors_dir + '.kl.test.val', model_type)
                            tm.generate_topic_dists(model_dir, vectors_dir + '.test.val', model_type)

    if infer_test_flag:
        test_base_dir = '../iclrlarge_dataset'
        for para in paragraphs:
            test_data_dir = test_base_dir + '/data_splits/dim.all.mod.neu.para.{}.test.val.text'.format(para)
            tm = TopicModels(test_data_dir, None)
            for dim in dimensions:
                for mode in dimensions[dim]:
                    for model_type in model_types:
                        for topic in topics:

                            train_data_dir = base_dir + '/data_splits/dim.{}.mod.{}.para.{}.train.text'.format(dim, mode, para)
                            model_dir = base_dir + '/lda_models/'
                            model_dir += '{}_topics/dim.{}.mod.{}.para.{}.num.{}/model'.format(topic, dim, mode, para, topic)
                            vectors_dir = test_base_dir + '/lda_vectors_{}/'.format(model_type)
                            vectors_dir += '{}_topics/dim.{}.mod.{}.para.{}.num.{}'.format(topic, dim, mode, para, topic)
                            tm.upload_vocab(train_data_dir)
                            print('infer ' + vectors_dir)

                            tm.generate_topic_kl(model_dir, vectors_dir + '.kl.test.val', model_type)


if __name__ == '__main__':
    main()
