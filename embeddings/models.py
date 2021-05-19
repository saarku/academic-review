from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dot, Activation, BatchNormalization, Dropout, Dense, Bidirectional, Lambda
import keras.backend as K
import numpy as np


class NeuralModel:
    def __init__(self, sequence_length=100, n_hidden=50):
        self.n_hidden = n_hidden
        self.sequence_length = sequence_length
        self.vocab_size = 1000

    def create_model(self, weights_dir=None):
        input_data = Input(shape=(self.sequence_length,), dtype='int32')

        embedding_layer = Embedding(self.vocab_size + 2, self.n_hidden, input_length=self.sequence_length, trainable=True)
        encoded = embedding_layer(input_data)

        lstm_layer = LSTM(self.n_hidden)
        lstm_output = lstm_layer(encoded)

        embedding_layer = Dense(1)
        outputs = embedding_layer(lstm_output)

        model = Model([input_data], outputs)

        if weights_dir is not None:
            model.load_weights(weights_dir)

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model


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


def load_data(data_dir, ids_dir, grades_dir, dimension, sequence_length=100, vocabulary_size=1000):
    labels_matrix = build_labels(ids_dir, grades_dir)
    labels = []
    vectors = []

    with open(data_dir, 'r') as vector_file:
        for i, line in enumerate(vector_file):
            label = labels_matrix[i, dimension]
            if label > 0:
                padding = [0] * sequence_length
                padded_line = [int(j) for j in line.split()] + padding
                padded_line = np.asarray(padded_line[0:sequence_length])
                padded_line[padded_line > vocabulary_size] = vocabulary_size + 1
                vectors.append(np.asarray(padded_line))
                labels.append(label)
    return np.vstack(vectors), np.asarray(labels)


def main():
    data_dir = '/home/skuzi2/education_dataset/embeddings_data/train.txt'
    ids_dir = '/home/skuzi2/education_dataset/data_splits/dim.all.mod.neu.para.1.train.ids'
    grades_dir = '/home/skuzi2/education_dataset/annotations/annotation_aggregated.tsv'

    model = NeuralModel()
    compiled_model = model.create_model()
    data, labels = load_data(data_dir, ids_dir, grades_dir, 1)
    print(data.shape)
    print(labels.shape)
    compiled_model.fit(x=data, y=labels, batch_size=16, epochs=3)
    compiled_model.save('lstm.hdf5')


if __name__ == '__main__':
    main()