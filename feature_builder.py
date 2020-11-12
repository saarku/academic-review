from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from topic_models import get_topics_vec
from utils import to_sparse, from_sparse, pre_process_text


class FeatureBuilder:

    def __init__(self, data_dir):
        self.count_vector = CountVectorizer(min_df=5)
        self.tf_idf_transformer = TfidfTransformer()
        train_data_dir = data_dir + '/train.text'
        test_data_dir = data_dir + '/test.val.text'
        grades_dir = data_dir + '/annotation_aggregated.tsv'
        self.train_labels = self.build_labels(data_dir + '/train.ids', grades_dir)
        self.test_labels = self.build_labels(data_dir + '/test.val.ids', grades_dir)
        self.train_lines = [pre_process_text(line) for line in open(train_data_dir, 'r').read().split('\n')][0:-1]
        self.test_lines = [pre_process_text(line) for line in open(test_data_dir, 'r').read().split('\n')][0:-1]

    def build_unigram_features(self, dimension_id):
        """ Build unigram features for a specific grading dimension.

        :param dimension_id: (int) id for the ranking dimension.
        :return: train/test features and labels.
        """
        train_data, y_train = self.modify_data_to_dimension(self.train_lines, self.train_labels, dimension_id)
        test_data, y_test = self.modify_data_to_dimension(self.test_lines, self.test_labels, dimension_id)
        x_train_counts = self.count_vector.fit_transform(train_data)
        x_test_counts = self.count_vector.transform(test_data)
        x_train_tf_idf = self.tf_idf_transformer.fit_transform(x_train_counts)
        x_test_tf_idf = self.tf_idf_transformer.transform(x_test_counts)
        y_train = np.asarray(y_train, dtype=float)
        return x_train_tf_idf, y_train, x_test_tf_idf, y_test

    def build_topic_features(self, dimension_id, topics_train_dir, topics_test_dir):
        x_train = get_topics_vec(topics_train_dir)
        x_test = get_topics_vec(topics_test_dir)
        train_data, y_train = self.modify_topics_to_dimension(x_train, self.train_labels, dimension_id)
        test_data, y_test = self.modify_topics_to_dimension(x_test, self.test_labels, dimension_id)
        return train_data, y_train, test_data, y_test

    @staticmethod
    def build_labels(ids_dir, grades_dir):
        """ Gather the labels for a specific data set.

        :param ids_dir: (string) the paper ids of the data set.
        :param grades_dir: (string) the grades of all papers.
        :return: Matrix. The labels of the papers in the different dimensions.
        """
        grade_lines = [line.split('\t') for line in open(grades_dir, 'r').read().split('\n')]
        grades_dict = {args[0]: args[1:len(args)] for args in grade_lines}
        ids = open(ids_dir, 'r').read().split('\n')[0:-1]
        num_dimensions = len(grade_lines[0]) - 1
        num_examples = len(ids)
        labels_matrix = np.zeros((num_examples, num_dimensions))

        for i, paper_id in enumerate(ids):
            for dimension_id in range(num_dimensions):
                grade_str = grades_dict[paper_id][dimension_id]
                grade = 0.0
                if grade_str != '-': grade = float(grade_str)
                labels_matrix[i, dimension_id] = grade
        return labels_matrix

    @staticmethod
    def modify_data_to_dimension(data_lines, grades_matrix, dimension_id):
        """ Modify a data set for papers with an actual grade in a dimension.

        :param data_lines: (list) the papers text.
        :param grades_matrix: (matrix) the grades of papers in different dimensions.
        :param dimension_id: (int) the dimension for the grade.
        :return: (list, list). Modified text and modified grades.
        """
        modified_lines = []
        modified_grades = []
        for i, line in enumerate(data_lines):
            grade = grades_matrix[i, dimension_id]
            if grade > 0:
                modified_grades.append(grade)
                modified_lines.append(line)
        return modified_lines, modified_grades

    @staticmethod
    def modify_topics_to_dimension(topics_matrix, grades_matrix, dimension_id):
        """ Modify a data set for papers with an actual grade in a dimension.

        :param data_lines: (list) the papers text.
        :param grades_matrix: (matrix) the grades of papers in different dimensions.
        :param dimension_id: (int) the dimension for the grade.
        :return: (list, list). Modified text and modified grades.
        """
        indexes = []
        modified_grades = []
        for i in range(topics_matrix.shape[0]):
            grade = grades_matrix[i, dimension_id]
            if grade > 0:
                modified_grades.append(grade)
                indexes.append(i)
        return topics_matrix[indexes, :], modified_grades