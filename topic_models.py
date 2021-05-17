from utils import to_sparse, from_sparse, pre_process_text
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import numpy as np
import joblib


def get_topics_vec(dists_dir, labels, dimension_id, num_paragraphs):
    """ Read topic distributions from a file and construct a sparse matrix.

    :param dists_dir: a file with the distributions.
    :param dimension_id: the dimension for grading.
    :param num_paragraphs: number of paragraphs.
    :return: csr_matrix.
    """
    all_topic_vectors = []
    distributions = open(dists_dir, 'r').readlines()
    for i in range(0, len(distributions), num_paragraphs):
        single_vec = {}
        for j in range(num_paragraphs):
            args = distributions[i+j].rstrip('\n').rstrip(']').lstrip('[').split('),')
            for a in args:
                index = int(a.split(',')[0].split('(')[1])
                number = float(a.split(',')[1].rstrip(')'))
                single_vec[index] = max(single_vec.get(index, 0), number)
        single_vec = list(single_vec.items())
        all_topic_vectors.append(single_vec)

    topic_vectors = []
    modified_grades = []
    for i, vector in enumerate(all_topic_vectors):
        grade = labels[i, dimension_id]
        if grade > 0:
            modified_grades.append(grade)
            topic_vectors.append(vector)
    num_topics = len(topic_vectors[0])
    return to_sparse(topic_vectors, (len(topic_vectors), num_topics)), modified_grades


class TopicModels:

    def __init__(self, data_dir, vocabulary_data_dir):
        self.data_lines = [pre_process_text(line) for line in open(data_dir, 'r').read().split('\n')][0:-1]
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

    def generate_topic_kl(self, topic_model_dir, output_dir):
        """ Generate topic representation (with kl-divergence) for training/test data.

        :param topic_model_dir: (string) directory of the topic model.
        :param output_dir: (string).
        :return: None.
        """
        lda_file = open(output_dir, 'w+')
        lda_model = LdaModel.load(topic_model_dir)
        count_vector_lda = CountVectorizer()
        count_vector_lda.fit(self.vocab_lines)
        x_lda_counts = count_vector_lda.transform(self.data_lines)
        x_vectors = from_sparse(x_lda_counts)
        all_topics = lda_model.get_topics()

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
        lda_file.close()


def main():

    topics = 5
    modes = ['pos', 'neg']
    dimensions = {'0': modes, '1': modes, '2': modes, '3': modes, '4': modes, '5': modes, '6': modes, 'all': ['neu']}
    paragraphs = ['1', '3']
    base_dir = '../education_dataset/'
    model_type = 'ovb' # ovb or gibbs

    learn_flag = False
    infer_flag = True

    if learn_flag:
        for dim in dimensions:
            for mode in dimensions[dim]:
                for para in paragraphs:
                    data_dir = base_dir + 'data_splits/dim.{}.mod.{}.para.{}.train.text'.format(dim, mode, para)
                    model_dir = base_dir + 'lda_models/{}_topics/'.format(topics)
                    model_dir += 'dim.{}.mod.{}.para.{}.num.{}/'.format(dim, mode, para, topics)
                    os.mkdir(model_dir) if not os.path.exists(model_dir) else None
                    print('learn ' + model_dir)
                    tm = TopicModels(data_dir, data_dir)
                    tm.learn_lda(topics, model_dir + '/model', model_type)

    if infer_flag:
        for dim in dimensions:
            for mode in dimensions[dim]:
                for para in paragraphs:
                    train_data_dir = base_dir + '/data_splits/dim.all.mod.neu.para.{}.train.text'.format(para)
                    test_data_dir = base_dir + '/data_splits/dim.all.mod.neu.para.{}.test.val.text'.format(para)
                    vocab_dir = base_dir + '/data_splits/dim.{}.mod.{}.para.{}.train.text'.format(dim, mode, para)
                    model_dir = base_dir + '/lda_models/'
                    vectors_dir = base_dir + '/lda_vectors_{}/'.format(model_type)
                    model_dir += '{}_topics/dim.{}.mod.{}.para.{}.num.{}/model'.format(topics, dim, mode, para, topics)
                    vectors_dir += '{}_topics/dim.{}.mod.{}.para.{}.num.{}'.format(topics, dim, mode, para, topics)
                    print('infer ' + vectors_dir)

                    tm = TopicModels(train_data_dir, vocab_dir)
                    tm.generate_topic_dists(model_dir, vectors_dir + '.train', model_type)
                    tm = TopicModels(test_data_dir, vocab_dir)
                    tm.generate_topic_dists(model_dir, vectors_dir + '.test.val', model_type)


if __name__ == '__main__':
    main()
