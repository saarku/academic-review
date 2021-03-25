from utils import to_sparse, from_sparse, pre_process_text
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer


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

    def __init__(self, data_dir):
        train_data_dir = data_dir + '/train.paragraphs.text'
        test_data_dir = data_dir + '/test.val.paragraphs.text'
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
    num_topics = 5

    tm = TopicModels(data_dir)

    tm.learn_lda(num_topics, '/home/skuzi2/iclr17_dataset/lda_models/' + str(num_topics) + '_topics/lda_para_' +
                 str(num_topics))
    tm.generate_topic_dists('/home/skuzi2/iclr17_dataset/lda_models/' + str(num_topics) + '_topics/lda_para_' +
                            str(num_topics), '/home/skuzi2/iclr17_dataset/lda_models/' + str(num_topics) +
                            '_topics/' + str(num_topics) + '_para_topics')


if __name__ == '__main__':
    main()
