from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from nltk import stem
from utils import pre_process_text
import sys


def process_data(data_dir):
    output_file = open(data_dir + '.processed', 'w')
    with open(data_dir, 'r') as input_file:
        for i, line in enumerate(input_file):
            if i % 1000 == 0: print(i)
            output_file.write(pre_process_text(line.rstrip('\n')) + '\n')
            output_file.flush()


class SearchEngine:

    def __init__(self, data_dir, ids_dir, aspect_dir):
        self.paper_ids = [i.rstrip('\n') for i in open(ids_dir, 'r').readlines()]
        self.vectors, self.names, self.tf_idf, self.counter = self.get_tf_idf_embeddings(data_dir)
        self.knn_engine = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='cosine').fit(self.vectors)
        self.aspects = self.load_aspects(aspect_dir)

    @staticmethod
    def get_tf_idf_embeddings(data_dir):
        data_lines = open(data_dir, 'r').readlines()
        count_vector = CountVectorizer()
        tf_vectors = count_vector.fit_transform(data_lines)
        tf_idf_transformer = TfidfTransformer()
        tf_idf_vectors = tf_idf_transformer.fit_transform(tf_vectors)
        return tf_idf_vectors, count_vector.get_feature_names(), tf_idf_transformer, count_vector

    @staticmethod
    def load_aspects(aspect_dir):
        lines = open(aspect_dir, 'r').readlines()
        aspects = lines[0].rstrip('\n').split(',')[1:]
        aspect_scores = {aspect: {} for aspect in aspects}

        for line in lines[1:]:
            args = line.rstrip('\n').split(',')
            paper_id = args[0]
            scores = args[1:]

            for i, score in enumerate(scores):
                aspect_scores[aspects[i]][paper_id] = float(score)
        return aspect_scores

    def search(self, query):
        top_words = {}
        query = pre_process_text(query)
        query = self.counter.transform([query])
        query = self.tf_idf.transform(query)
        distances, neighbor_indexes = self.knn_engine.kneighbors(query)

        result_list = []
        for i in range(len(neighbor_indexes[0])):
            result_list.append(self.paper_ids[neighbor_indexes[0][i]])

        top_words['Relevance'] = self.get_top_words(result_list)

        for aspect in self.aspects:
            sorted_list = self.re_rank(result_list, aspect)
            top_words[aspect] = self.get_top_words(sorted_list)
        return top_words

    def re_rank(self, result_list, aspect):
        result_scores = {}
        for paper_id in result_list:
            result_scores[paper_id] = self.aspects[aspect][paper_id]
        return sorted(result_scores, key=result_scores.get, reverse=True)

    def get_top_words(self, result_list, num_words=20):
        stopwords = ['de', 'et', 'al', 'une', 'la', 'po', 'le', 'use']
        avg_vec = np.zeros((1, self.vectors.shape[1]))
        weights = [1/float(i+1) for i in range(10)]
        normalizer = sum(weights)

        for i, paper_id in enumerate(result_list[:10]):
            paper_idx = self.paper_ids.index(paper_id)
            avg_vec += (weights[i]/normalizer) * self.vectors[paper_idx, :]

        avg_vec /= len(result_list)
        word_id_dict = {}
        for i in range(avg_vec.shape[1]):
            word_id_dict[i] = avg_vec[0, i]
        sorted_words = sorted(word_id_dict, key=word_id_dict.get, reverse=True)
        top_words = []

        for i in range(num_words*2):
            word = self.names[sorted_words[i]]
            if word not in stopwords:
                top_words.append(word)

        return top_words[:20]

    def analyze_queries(self, queries):
        output_file = open('queries.txt', 'w')
        for q in queries:
            top_words = self.search(q)
            for aspect in top_words:
                output_file.write(q + ',' + aspect + ',' + ','.join(top_words[aspect]) + '\n')
                output_file.flush()


def main():
    query = ['language model', 'lda', 'word embeddings']
    data_dir = '/home/skuzi2/acl_dataset/data_splits/dim.all.mod.neu.para.1.test.val'
    aspects_dir = '/home/skuzi2/academic-review/acl_aspects.txt'
    se = SearchEngine(data_dir + '.text.processed', data_dir + '.ids', aspects_dir)
    se.analyze_queries(query)


if __name__ == '__main__':
    main()
