from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from utils import pre_process_text
import numpy as np


class SearchEngine:

    def __init__(self, data_dir, ids_dir):
        self.paper_ids = [i.rstrip('\n') for i in open(ids_dir, 'r').readlines()]
        self.vectors, self.names, self.tf_idf, self.counter = self.get_tf_idf_embeddings(data_dir)
        self.knn_engine = NearestNeighbors(n_neighbors=50 + 1, algorithm='brute', metric='cosine').fit(self.vectors)

    @staticmethod
    def get_tf_idf_embeddings(data_dir, num_features=1000):
        data_lines = []

        print('read')
        with open(data_dir, 'r') as input_file:
            for i, line in enumerate(input_file):
                data_lines.append(line.rstrip('\n'))
        print('finished')

        count_vector = CountVectorizer(max_features=num_features)
        tf_vectors = count_vector.fit_transform(data_lines)
        tf_idf_transformer = TfidfTransformer()
        tf_idf_vectors = tf_idf_transformer.fit_transform(tf_vectors)
        return tf_idf_vectors, count_vector.get_feature_names(), tf_idf_transformer, count_vector

    def search(self, query):
        query = pre_process_text(query)
        query = self.counter.transform([query])
        query = self.tf_idf.transform(query)
        distances, neighbor_indexes = self.knn_engine.kneighbors(query)
        result_list = []
        for i in range(len(neighbor_indexes[0])):
            result_list.append(self.paper_ids[i])
        self.get_top_words(result_list)

    def get_top_words(self, result_list):
        avg_vec = np.zeros((1, self.vectors.shape[1]))

        for paper_id in result_list:
            paper_idx = self.paper_ids.index(paper_id)
            avg_vec += self.vectors[paper_idx, :]

        avg_vec /= len(result_list)
        word_id_dict = {}
        for i in range(avg_vec.shape[1]):
            word_id_dict[i] = avg_vec[0, i]
        sorted_words = sorted(word_id_dict, key=word_id_dict.get, reverse=True)
        for i in range(10):
            print(self.names[sorted_words[i]])


def main():
    query = 'neural network'
    data_dir = '/home/skuzi2/acl_dataset/data_splits/dim.all.mod.neu.para.1.test.val'
    se = SearchEngine(data_dir + '.text', data_dir + '.ids')
    se.search(query)



if __name__ == '__main__':
    main()
