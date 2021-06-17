from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from utils import pre_process_text


class SearchEngine:

    def __init__(self, data_dir, ids_dir):
        paper_ids = [i.rstrip('\n') for i in open(ids_dir, 'r').readlines()]
        vectors, names, self.tf_idf, self.counter = self.get_tf_idf_embeddings(data_dir)
        self.knn_engine = NearestNeighbors(n_neighbors=50 + 1, algorithm='brute', metric='cosine').fit(vectors)

    @staticmethod
    def get_tf_idf_embeddings(data_dir, num_features=1000):
        data_lines = [pre_process_text(line.rstrip('\n')) for line in open(data_dir, 'r').readlines()]
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
        print(neighbor_indexes)


def main():
    query = 'neural network'
    data_dir = '/home/skuzi2/acl_dataset/data_splits/dim.all.mod.neu.para.1.test'
    se = SearchEngine(data_dir + '.text', data_dir + '.ids')
    se.search(query)


if __name__ == '__main__':
    main()
