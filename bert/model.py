import tensorflow as tf
from bert_layer import BertLayer
from keras.models import Model
import keras.backend as K
sess = tf.Session()


def triplet_distance(vectors):
    query_vector, positive_vector, negative_vector = vectors
    sim_pos = K.dot(query_vector, K.transpose(positive_vector))
    sim_neg = K.dot(query_vector, K.transpose(negative_vector))
    return sim_pos - sim_neg


class BertModel:
    def __init__(self, sequence_length=20, n_hidden=50):
        self.n_hidden = n_hidden
        self.sequence_length = sequence_length
        self.dropout_rate = 0.5

    def create_model(self, losses, weights_dir=None):

        in_id_left = tf.keras.layers.Input(shape=(self.sequence_length,), name="input_ids_l")
        in_mask_left = tf.keras.layers.Input(shape=(self.sequence_length,), name="input_masks_l")
        in_segment_left = tf.keras.layers.Input(shape=(self.sequence_length,), name="segment_ids_l")
        left_input = [in_id_left, in_mask_left, in_segment_left]

        in_id_right = tf.keras.layers.Input(shape=(self.sequence_length,), name="input_ids_r")
        in_mask_right = tf.keras.layers.Input(shape=(self.sequence_length,), name="input_masks_r")
        in_segment_right = tf.keras.layers.Input(shape=(self.sequence_length,), name="segment_ids_r")
        right_input = [in_id_right, in_mask_right, in_segment_right]

        bert_layer = BertLayer(n_fine_tune_layers=1, pooling="first")
        bert_left = bert_layer(left_input)
        bert_right = bert_layer(right_input)

        dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        left_output = dropout_layer(bert_left)
        right_output = dropout_layer(bert_right)

        '''
        bn_layer = tf.keras.layers.BatchNormalization()
        left_output = bn_layer(left_output)
        right_output = bn_layer(right_output)
        '''

        dense_layer = tf.keras.layers.Dense(100, activation='relu')
        left_output = dense_layer(left_output)
        right_output = dense_layer(right_output)

        dense_layer = tf.keras.layers.Dense(50)
        left_output = dense_layer(left_output)
        right_output = dense_layer(right_output)

        dot_layer_1 = tf.keras.layers.Dot(1)
        dot_layer_output = dot_layer_1([left_output, right_output])
        sigmoid_layer = tf.keras.layers.Activation(activation='sigmoid')
        distance_1 = sigmoid_layer(dot_layer_output)

        dot_layer_2 = tf.keras.layers.Dot(1)
        distance_2 = dot_layer_2([left_output, right_output])

        outputs = []
        training_losses = []
        weights = []
        if 'bin' in losses:
            outputs += [distance_1]
            training_losses += ['binary_crossentropy']
            weights += [1]
        if 'multi' in losses:
            outputs += [distance_2]
            training_losses += ['mse']
            weights += [1]

        model = tf.keras.Model([in_id_left, in_mask_left, in_segment_left, in_id_right, in_mask_right,
                                in_segment_right], outputs)

        if weights_dir is not None:
            model.load_weights(weights_dir)

        model.compile(loss=training_losses, optimizer='adam', metrics=['accuracy'], loss_weights=weights)

        print(model.summary())

        return model

    def create_hinge_model(self, weights_dir=None):
        in_id_query = tf.keras.layers.Input(shape=(self.sequence_length,), name="input_ids_q")
        in_mask_query = tf.keras.layers.Input(shape=(self.sequence_length,), name="input_masks_q")
        in_segment_query = tf.keras.layers.Input(shape=(self.sequence_length,), name="segment_ids_q")
        query_input = [in_id_query, in_mask_query, in_segment_query]

        in_id_pos = tf.keras.layers.Input(shape=(self.sequence_length,), name="input_ids_p")
        in_mask_pos = tf.keras.layers.Input(shape=(self.sequence_length,), name="input_masks_p")
        in_segment_pos = tf.keras.layers.Input(shape=(self.sequence_length,), name="segment_ids_p")
        pos_input = [in_id_pos, in_mask_pos, in_segment_pos]

        in_id_neg = tf.keras.layers.Input(shape=(self.sequence_length,), name="input_ids_n")
        in_mask_neg = tf.keras.layers.Input(shape=(self.sequence_length,), name="input_masks_n")
        in_segment_neg = tf.keras.layers.Input(shape=(self.sequence_length,), name="segment_ids_n")
        neg_input = [in_id_neg, in_mask_neg, in_segment_neg]

        bert_layer = BertLayer(n_fine_tune_layers=1, pooling="first")
        bert_query = bert_layer(query_input)
        bert_pos = bert_layer(pos_input)
        bert_neg = bert_layer(neg_input)

        dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        query_output = dropout_layer(bert_query)
        neg_output = dropout_layer(bert_pos)
        pos_output = dropout_layer(bert_neg)

        dense_layer = tf.keras.layers.Dense(100, activation='relu')
        query_output = dense_layer(query_output)
        pos_output = dense_layer(pos_output)
        neg_output = dense_layer(neg_output)

        dense_layer_2 = tf.keras.layers.Dense(50)
        query_output = dense_layer_2(query_output)
        pos_output = dense_layer_2(pos_output)
        neg_output = dense_layer_2(neg_output)

        distance_difference = tf.keras.layers.Lambda(triplet_distance)([query_output, pos_output, neg_output])
        model = tf.keras.Model(inputs=[query_input, pos_input, neg_input], outputs=[distance_difference])

        if weights_dir is not None:
            model.load_weights(weights_dir)

        model.compile(loss='hinge', optimizer='adam')
        print(model.summary())
        return model