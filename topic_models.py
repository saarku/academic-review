from feature_builder import to_sparse, from_sparse, pre_process_text
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer


def get_topics_vec(dists_dir):
    """ Read topic distributions from a file and construct a sparse matrix.

    :param dists_dir: a file with the distributions.
    :return: csr_matrix.
    """
    topic_vectors = []
    with open(dists_dir, 'r') as input_file:
        for line in input_file:
            args = line.rstrip('\n').rstrip(']').lstrip('[').split('),')
            single_vec = []
            for a in args:
                index = int(a.split(',')[0].split('(')[1])
                number = float(a.split(',')[1].rstrip(')'))
                single_vec += [(index, number)]
            topic_vectors += [single_vec]
        num_topics = len(topic_vectors[0])
    return to_sparse(topic_vectors, (len(topic_vectors), num_topics))


class TopicModels:

    def __init__(self, data_dir):
        train_data_dir = data_dir + '/train.text'
        test_data_dir = data_dir + '/test.val.text'
        self.train_lines = [pre_process_text(line) for line in open(train_data_dir, 'r').read().split('\n')][0:-1]
        self.test_lines = [pre_process_text(line) for line in open(test_data_dir, 'r').read().split('\n')][0:-1]

    def learn_lda(self, num_topics, output_dir):
        """ Learn an LDA topic model.

        :param num_topics: (int) number file with the distributions.
        :param output_dir: (string) directory for the output model.
        :return: None.
        """
        count_vector = CountVectorizer()
        train_data = count_vector.fit_transform(self.train_lines)
        train_data = from_sparse(train_data)
        lda = LdaModel(train_data, num_topics=num_topics)
        lda.save(output_dir)

    def generate_topic_dists(self, topic_model_dir, output_dir):
        """ Generate topic representation for training/test data.

        :param topic_model_dir: (string) directory of the topic model.
        :param output_dir: (string).
        :return: None.
        """
        lda_model = LdaModel.load(topic_model_dir)
        count_vector_lda = CountVectorizer()
        count_vector_lda.fit(self.train_lines)
        x_train_lda_counts = count_vector_lda.transform(self.train_lines)
        x_test_lda_counts = count_vector_lda.transform(self.test_lines)
        train_vectors = from_sparse(x_train_lda_counts)
        test_vectors = from_sparse(x_test_lda_counts)
        lda_train_file = open(output_dir + '.train', 'w+')
        lda_test_file = open(output_dir + '.test', 'w+')
        for lda_vector in train_vectors:
            lda_train_file.write(str(lda_model.get_document_topics(lda_vector, minimum_probability=0.0)) + '\n')
        for lda_vector in test_vectors:
            lda_test_file.write(str(lda_model.get_document_topics(lda_vector, minimum_probability=0.0)) + '\n')
        lda_train_file.close()
        lda_test_file.close()


def main():
    data_dir = '/home/skuzi2/iclr17_dataset'
    tm = TopicModels(data_dir)

    tm.learn_lda(20, '/home/skuzi2/iclr17_dataset/lda_models/20_topics/lda_20')
    tm.generate_topic_dists('/home/skuzi2/iclr17_dataset/lda_models/20_topics/lda_20',
                            '/home/skuzi2/iclr17_dataset/lda_models/20_topics/20_topics')


if __name__ == '__main__':
    main()
