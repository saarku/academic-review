from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dot, Flatten, Activation, BatchNormalization, Dropout, Dense, Bidirectional, Lambda, MaxPooling1D, Conv1D
import numpy as np
import sys
from tensorflow.keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping


class NeuralModel:
    def __init__(self, sequence_length=100, n_hidden=50, embedding_dim=50):
        self.n_hidden = n_hidden
        self.sequence_length = sequence_length
        self.vocab_size = 1000
        self.embedding_dim = embedding_dim

    def create_model(self, weights_dir=None, optimizer='adam'):
        input_data = Input(shape=(self.sequence_length,), dtype='int32')

        embedding_layer = Embedding(self.vocab_size + 2, self.embedding_dim, input_length=self.sequence_length,
                                    trainable=True)
        encoded = embedding_layer(input_data)

        lstm_layer = LSTM(self.n_hidden)
        lstm_output = lstm_layer(encoded)

        embedding_layer = Dense(1)
        outputs = embedding_layer(lstm_output)

        model = Model([input_data], outputs)

        if weights_dir is not None:
            model.load_weights(weights_dir)

        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return model

    def create_cnn(self, weights_dir=None, optimizer='adam'):
        input_data = Input(shape=(self.sequence_length,), dtype='int32')
        embedding_layer = Embedding(self.vocab_size + 2, self.embedding_dim, input_length=self.sequence_length,
                                    trainable=True)
        encoded = embedding_layer(input_data)
        conv_layer = Conv1D(4, kernel_size=3, activation='relu', input_shape=(self.sequence_length,
                                                                                   self.embedding_dim))
        cnn_output = conv_layer(encoded)

        pooling_layer = MaxPooling1D()
        cnn_output = pooling_layer(cnn_output)

        flatten_layer = Flatten()
        cnn_output = flatten_layer(cnn_output)

        dense_layer_1 = Dense(self.n_hidden)
        dense_layer_2 = Dense(1)

        outputs = dense_layer_1(cnn_output)
        outputs = dense_layer_2(outputs)

        model = Model([input_data], outputs)

        if weights_dir is not None:
            model.load_weights(weights_dir)

        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
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


def load_data(data_dir, ids_dir, grades_dir, dimension, sequence_length=100, vocabulary_size=1000, infer_flag=False):
    labels_matrix = build_labels(ids_dir, grades_dir)
    labels = []
    vectors = []

    with open(data_dir, 'r') as vector_file:
        for i, line in enumerate(vector_file):
            label = labels_matrix[i, dimension]
            if label > 0 or infer_flag:
                padding = [0] * sequence_length
                padded_line = [int(j) for j in line.split()] + padding
                padded_line = np.asarray(padded_line[0:sequence_length])
                padded_line[padded_line > vocabulary_size] = vocabulary_size + 1
                vectors.append(np.asarray(padded_line))
                labels.append(label)
    return np.vstack(vectors), np.asarray(labels)


def train_model(data_name, grades_dim, dimension=20, w_dimension=50, epochs=3, batch_size=16, optimizer='adam',
                vocab=1000, length=100):
    base_dir = '/home/skuzi2/{}_dataset/'.format(data_name)
    data_dir = base_dir + '/embeddings_data/train.txt'
    ids_dir = base_dir + '/data_splits/dim.all.mod.neu.para.1.train.ids'
    grades_dir = base_dir + '/annotations/annotation_aggregated.tsv'
    model_name = 'cnn.dim.{}.ldim.{}.wdim.{}.epoch.{}.batch.{}.opt.{}.vocab.{}.length.{}'.format(grades_dim, dimension,
                                                                                                  w_dimension, epochs,
                                                                                                  batch_size, optimizer,
                                                                                                  vocab, length)
    model_dir = base_dir + '/embeddings_models/' + model_name + '.hdf5'
    model = NeuralModel(n_hidden=dimension, embedding_dim=w_dimension, sequence_length=length)

    #compiled_model = model.create_model(optimizer=optimizer)
    compiled_model = model.create_cnn(optimizer=optimizer)
    data, labels = load_data(data_dir, ids_dir, grades_dir, grades_dim, vocabulary_size=vocab, sequence_length=length)
    batch_size = min(batch_size, data.shape[0])

    #early_stop = EarlyStopping(monitor='val_loss', restore_best_weights=True)
    #model_checkpoint = ModelCheckpoint(filepath=model_dir, monitor='val_loss', save_best_only=True)
    #callbacks_list = [model_checkpoint, early_stop]
    compiled_model.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs)#, validation_split=0.2) # callbacks=callbacks_list)
    compiled_model.save(model_dir)
    return model_name


def infer_embeddings(data_name, grades_dim, model_name, data_type, vocab=1000, length=100):
    K.clear_session()
    base_dir = '/home/skuzi2/{}_dataset/'.format(data_name)
    data_dir = base_dir + '/embeddings_data/{}.txt'.format(data_type)
    ids_dir = base_dir + '/data_splits/dim.all.mod.neu.para.1.{}.ids'.format(data_type)
    grades_dir = base_dir + '/annotations/annotation_aggregated.tsv'
    model_dir = base_dir + '/embeddings_models/' + model_name + '.hdf5'
    vectors_dir = base_dir + '/embeddings_vectors/' + model_name + '.' + data_type
    model_args = model_name.split('.')

    model = NeuralModel(n_hidden=int(model_args[4]), embedding_dim=int(model_args[6]), sequence_length=length)

    #compiled_model = model.create_model(weights_dir=model_dir)
    compiled_model = model.create_cnn(weights_dir=model_dir)

    print(compiled_model.summary())
    #intermediate_layer_model = Model(inputs=compiled_model.get_layer('input_1').output,
    #                                 outputs=compiled_model.get_layer('lstm').get_output_at(0))

    intermediate_layer_model = Model(inputs=compiled_model.get_layer('input_1').output,
                                     outputs=compiled_model.get_layer('dense').get_output_at(0))

    data, _ = load_data(data_dir, ids_dir, grades_dir, grades_dim, infer_flag=True, vocabulary_size=vocab,
                        sequence_length=length)
    embeddings = intermediate_layer_model.predict(data)

    with open(vectors_dir, 'w+') as output_file:
        for line_num in range(embeddings.shape[0]):
            output_file.write('[' + ', '.join(['(' + str(i) + ', ' + str(num) + ')'
                                              for i, num in enumerate(list(embeddings[line_num, :]))]) + ']\n')


def main():

    data_name = sys.argv[1]
    dimensions = [5, 15, 25]
    w_dims = [20]
    epochs = [5]
    batch_sizes = [16]
    vocabs = [1000]
    lengths = [100]
    optimizers = ['adam']
    grade_dims = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]

    for grade_dim in grade_dims:
        for lstm_dim in dimensions:
            for word_dim in w_dims:
                for epoch in epochs:
                    for batch in batch_sizes:
                        for opt in optimizers:
                            for v in vocabs:
                                for l in lengths:
                                    model_name = train_model(data_name, grade_dim, dimension=lstm_dim,
                                                             w_dimension=word_dim, epochs=epoch, batch_size=batch,
                                                             optimizer=opt, vocab=v, length=l)
                                    infer_embeddings(data_name, grade_dim, model_name, 'train', vocab=v, length=l)
                                    infer_embeddings(data_name, grade_dim, model_name, 'test.val', vocab=v, length=l)


if __name__ == '__main__':
    main()