from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from nltk import stem
from utils import pre_process_text
from scipy.stats import ttest_rel
import sys
import os
import numpy as np
from scipy.stats import kendalltau, pearsonr
from collections import defaultdict
import re


def get_titles(data_dir):
    html_converter = re.compile(r'<[^>]+>')
    output_file = open('iclrlarge_titles.txt', 'w')
    for file_name in os.listdir(data_dir):
        paper_id = file_name.split('.')[0]
        with open(data_dir + '/' + file_name, 'r') as input_file:
            for line in input_file:
                if '<title>' in line:
                    t = html_converter.sub('', line.rstrip('\n')).lower()
                    t = t.replace('\n', ' ')
                    if len(t) > 0:
                        output_file.write(paper_id + ',' + t + '\n')


def filter_queries(queries_dir):
    data_dir = '/home/skuzi2/acl_dataset/data_splits/dim.all.mod.neu.para.1.test.val'
    aspects_dir = '/home/skuzi2/acl_dataset/acl_aspects.txt'
    citations_dir = '/home/skuzi2/acl_dataset/citation_counts.txt'
    years_dir = '/home/skuzi2/acl_dataset/years.txt'
    titles_dir = '/home/skuzi2/acl_dataset/acl_titles.txt'

    se = SearchEngine(data_dir + '.text.lemmarize', data_dir + '.ids', aspects_dir, citations_dir, titles_dir, years_dir)
    queries = [line.rstrip('\n') for line in open(queries_dir, 'r').readlines()]
    filtered = []
    for q in queries:
        num_terms = len(q.split())
        query = pre_process_text(q, lemmatize=True)
        query = se.counter.transform([query])
        query = se.tf_idf.transform(query)
        if query.count_nonzero() != num_terms: continue
        distances, neighbor_indexes = se.knn_engine.kneighbors(query)
        citations = []
        for i in range(len(neighbor_indexes[0])):
            paper_id = se.paper_ids[neighbor_indexes[0][i]]
            citations.append(se.citations['Citations'].get(paper_id, 0))
        if sum(citations) <= 0: continue
        filtered.append(q)
    output_file = open('filtered_queries.txt', 'w')
    for q in filtered: output_file.write(q + '\n')


def years_graph():
    data_name = sys.argv[1]
    query = ' '.join(sys.argv[2:len(sys.argv)])
    data_dir = '/home/skuzi2/{}_dataset/data_splits/dim.all.mod.neu.para.1.test.val'.format(data_name)
    aspects_dir = '/home/skuzi2/{}_dataset/{}_aspects.txt'.format(data_name, data_name)
    citations_dir = '/home/skuzi2/{}_dataset/citation_counts.txt'.format(data_name)
    titles_dir = '/home/skuzi2/{}_dataset/{}_titles.txt'.format(data_name, data_name)
    years_dir = '/home/skuzi2/{}_dataset/years.txt'.format(data_name, data_name)

    scores = defaultdict(list)
    years = ['0' + str(i) for i in range(10)] + [str(i) for i in range(10, 18)]
    years = ['17', '18', '19', '20']

    for y in years:
        print(y)
        se = SearchEngine(data_dir + '.text.lemmatize', data_dir + '.ids', aspects_dir, citations_dir, titles_dir,
                          years_dir, filter_flag=False, years_flag=y)
        count = se.years_analysis(query, y)
        for aspect in count:
            scores[aspect].append(count[aspect])

    output_file = open('years_scores_{}.txt'.format('_'.join(query.split())), 'w+')
    for aspect in scores:
        output_file.write(aspect + ','+ ','.join([str(i) for i in scores[aspect]]) + '\n')


def process_data(data_dir):
    output_file = open(data_dir + '.lemmatize', 'w')
    with open(data_dir, 'r') as input_file:
        for i, line in enumerate(input_file):
            output_file.write(pre_process_text(line.rstrip('\n'), lemmatize=True) + '\n')
            output_file.flush()


def analyze_evaluations(eval_dir):
    evals = defaultdict(dict)
    lines = open(eval_dir, 'r').readlines()
    performance = []
    for line in lines[1:]:
        args = line.rstrip('\n').split(',')
        qid, aspect, cutoff, value = args[0], args[2], args[4], float(args[5])
        if cutoff == '10':
            evals[qid][aspect] = value

    aspect_counters = defaultdict(int)
    aspect_counters_improve = defaultdict(int)
    for qid in evals:
        a = max(evals[qid], key=evals[qid].get)
        performance.append(evals[qid][a])
        aspect_counters[a] += 1
        for aspect in aspect_counters:
            diff = evals[qid][aspect] - evals[qid]['Relevance']
            if diff == 0:
                aspect_counters_improve[aspect] += 1

    print(aspect_counters)
    print(np.mean(performance))
    print(aspect_counters_improve)


def robustness_evaluations(eval_dir):
    evals = defaultdict(dict)
    lines = open(eval_dir, 'r').readlines()
    for line in lines[1:]:
        args = line.rstrip('\n').split(',')
        try:
            qid, aspect, cutoff, value = args[1], args[2], args[4], float(args[5])
            if cutoff == '10':
                evals[qid][aspect] = value
        except:
            continue


    for qid in evals:
        selected_aspects = []
        sorted_values = sorted(evals[qid], key=evals[qid].get, reverse=True)
        max_val = evals[qid][sorted_values[0]]
        for aspect in evals[qid]:
            if evals[qid][aspect] == max_val:
                selected_aspects.append(aspect)
        if len(selected_aspects) == 1:
            print(qid + ',' + ','.join(selected_aspects))

    '''
    histogram = {}
    histogram_q = defaultdict(set)
    for qid in evals:
        for aspect in evals[qid]:
            if aspect == 'Relevance': continue
            if aspect not in histogram:
                histogram[aspect] = {(-float('inf'), -20): 0, (-20, -10): 0, (-10, 0): 0, (0, 0.0000001): 0,
                                     (0.0000001, 10): 0, (10, 20): 0, (20, float('inf')): 0}

            diff = 100* (evals[qid][aspect] - evals[qid]['Relevance']) / (0.00000000001 + evals[qid]['Relevance'])
            if diff > 0:
                histogram_q[aspect].add(qid)
            for key in histogram[aspect]:
                if key[1] >= diff >= key[0]:
                    histogram[aspect][key] += 1

    for aspect in histogram:
        for key in histogram[aspect]:
            print(aspect + ',' + str(key[0]) + ',' + str(key[1]) + ',' + str(histogram[aspect][key]))

    for aspect in histogram:
        unique_queries = histogram_q[aspect]
        for other_aspect in histogram:
            if other_aspect != aspect:
                unique_queries = unique_queries - histogram_q[other_aspect]
        print('{},{},{}'.format(aspect, len(histogram_q[aspect]), len(unique_queries)))
    '''

class SearchEngine:

    def __init__(self, data_dir, ids_dir, aspect_dir, citation_dir, titles_dir, years_dir, filter_flag=True,
                 years_flag=''):
        self.citations = self.load_aspects(citation_dir)
        self.titles = self.load_titles(titles_dir)
        self.years = self.load_titles(years_dir)
        self.paper_ids = [i.rstrip('\n') for i in open(ids_dir, 'r').readlines()]
        if filter_flag:
            self.paper_ids = [i for i in self.paper_ids if i in self.citations['Citations']]
        if len(years_flag) > 0:
            self.paper_ids = [i for i in self.paper_ids if self.years.get(i, '') == years_flag]

        self.vectors, self.names, self.tf_idf, self.counter = self.get_tf_idf_embeddings(data_dir, ids_dir, filter_flag,
                                                                                         years_flag)
        print(self.vectors.shape)

        self.knn_engine = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='cosine').fit(self.vectors)
        self.aspects = self.load_aspects(aspect_dir)

    def get_tf_idf_embeddings(self, data_dir, ids_dir, filter_flag, years_flag):
        data_lines = open(data_dir, 'r').readlines()
        paper_ids = [i.rstrip('\n') for i in open(ids_dir, 'r').readlines()]

        if filter_flag:
            filtered_lines = []
            for i, line in enumerate(data_lines):
                if paper_ids[i] in self.citations['Citations']:
                    filtered_lines.append(line)
            data_lines = filtered_lines

        if len(years_flag):
            filtered_lines = []
            for i, line in enumerate(data_lines):
                if self.years.get(paper_ids[i], '') == years_flag:
                    filtered_lines.append(line)
            data_lines = filtered_lines

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

    @staticmethod
    def load_titles(aspect_dir):

        lines = open(aspect_dir, 'r').readlines()
        all_titles = {}
        for line in lines:
            args = line.rstrip('\n').split(',')
            paper_id = args[0]
            if aspect_dir is not None:
                title = ','.join(args[1:])
                title = '_'.join(title.split())
            all_titles[paper_id] = title
        return all_titles

    def search(self, query):
        top_words = {}
        top_scores = defaultdict(list)
        correlations = {}
        query = pre_process_text(query, lemmatize=True)
        query = self.counter.transform([query])
        query = self.tf_idf.transform(query)
        distances, neighbor_indexes = self.knn_engine.kneighbors(query)

        result_list = []
        for i in range(len(neighbor_indexes[0])):
            result_list.append(self.paper_ids[neighbor_indexes[0][i]])

        top_words['Relevance'] = self.get_top_words(result_list)
        correlations['Relevance'] = self.get_correlation(result_list)

        for aspect in self.aspects:
            sorted_list = self.re_rank(result_list, aspect)
            for paper_id in result_list[:50]:
                top_scores[aspect].append(self.aspects[aspect][paper_id])
            top_words[aspect] = self.get_top_words(sorted_list)
            correlations[aspect] = self.get_correlation(sorted_list)

        return top_words, correlations, top_scores

    def re_rank(self, result_list, aspect):
        result_scores = {}
        for paper_id in result_list:
            result_scores[paper_id] = self.aspects[aspect][paper_id]
        return sorted(result_scores, key=result_scores.get, reverse=True)

    def get_top_words(self, result_list, num_words=20):
        stopwords = ['de', 'et', 'al', 'une', 'la', 'po', 'le', 'use', 'aa', 'sc']
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

    def get_correlation(self, result_list):
        ranks, citations = [], []
        for i, paper_id in enumerate(result_list):
            ranks.append(1/float(i+1))
            citations.append(self.citations['Citations'].get(paper_id, 0))
        print(citations)
        kendall, _ = kendalltau(ranks, citations)

        return kendall, np.mean(citations[:5])

    def analyze_queries(self, queries):
        output_file = open('queries.txt', 'w')
        for q in queries:
            top_words, correlations, top_scores = self.search(q)
            for aspect in top_words:
                output_file.write(q + ',' + aspect + ',' + str(correlations[aspect]) + ',' + ','.join(top_words[aspect]) + '\n')
                #sorted_scores = sorted(top_scores[aspect])
                #output_file.write(q + ',' + aspect + ',' + ','.join([str(i) for i in sorted_scores]) + '\n')
                #output_file.flush()

    def years_analysis(self, query, y):
        top_words, correlations, top_scores = self.search(query)
        scores = {}
        for aspect in top_words:
            if aspect == 'Relevance': continue
            assert len(top_scores[aspect]) >= 10, 'len:{},{}'.format(len(top_scores[aspect]), y)
            scores[aspect] = np.mean(top_scores[aspect][:10])
        return scores

    @staticmethod
    def get_dcg(labels):
        relevance_dict = {(0, 0): 0, (1, 5): 1, (6, 10): 2, (11, 20): 3, (21, 100000): 4}
        dcg = 0
        for i, l in enumerate(labels):
            relevancy = 0
            for key in relevance_dict:
                if key[1] >= l >= key[0]:
                    relevancy = relevance_dict[key]
            dcg += (np.power(2, relevancy) - 1) / np.log2(i+2)
        return dcg

    def get_citation_dcg(self, result_list, cutoff):
        citations = []
        for i, paper_id in enumerate(result_list[:cutoff]):
            citations.append(self.citations['Citations'].get(paper_id, 0))
        dcg = self.get_dcg(citations)
        avg_citations = np.mean(citations)
        num_papers = sum([1 for c in citations if c>0])

        citations = []
        for paper_id in result_list:
            citations.append(int(self.citations['Citations'].get(paper_id, 0)))
        citations = sorted(citations, reverse=True)[:cutoff]
        ideal_dcg = self.get_dcg(citations)
        return dcg/ideal_dcg, avg_citations, num_papers

    def run_dataset(self, queries_dir):
        queries = [q.rstrip('\n') for q in open(queries_dir, 'r').readlines()]
        evaluations = defaultdict(dict)
        output_file = open('eval.txt', 'w')

        for qid, q in enumerate(queries):
            print(q)
            result_lists = defaultdict(list)
            query = pre_process_text(q, lemmatize=True)
            query = self.counter.transform([query])
            query = self.tf_idf.transform(query)
            distances, neighbor_indexes = self.knn_engine.kneighbors(query)

            for i in range(len(neighbor_indexes[0])):
                result_lists['Relevance'].append(self.paper_ids[neighbor_indexes[0][i]])
            for aspect in self.aspects:
                result_lists[aspect] = self.re_rank(result_lists['Relevance'], aspect)

            for aspect in result_lists:
                titles = []
                for paper_id in result_lists[aspect][:10]:
                    titles.append('_'.join(self.titles.get(paper_id, '').split()))
                output_file.write('{},{},{},{},{},{}\n'.format(qid, q, aspect, 'titles', 10, '$'.join(titles)))

                for k in [3, 5, 10]:
                    dcg, avg_citations, num_papers = self.get_citation_dcg(result_lists[aspect], k)

                    output_file.write('{},{},{},{},{},{}\n'.format(qid, q, aspect, 'ndcg', k, dcg))
                    output_file.write('{},{},{},{},{},{}\n'.format(qid, q, aspect, 'avgcite', k, avg_citations))
                    output_file.write('{},{},{},{},{},{}\n'.format(qid, q, aspect, 'avgpaper', k, num_papers))
                    if k not in evaluations[aspect]: evaluations[aspect][k] = []
                    evaluations[aspect][k].append(dcg)

        for aspect in evaluations:
            for k in evaluations[aspect]:
                _, p_val = ttest_rel(evaluations['Relevance'][k], evaluations[aspect][k])
                output_file.write('{},{},{},{},{},{}\n'.format('all', 'all', aspect, 'ndcgpval', k, p_val))


    def run_jaccard(self, queries):
        jaccard_dict = defaultdict(list)

        for qid, q in enumerate(queries):
            print(q)
            top_words = {}
            query = pre_process_text(q, lemmatize=True)
            query = self.counter.transform([query])
            query = self.tf_idf.transform(query)
            distances, neighbor_indexes = self.knn_engine.kneighbors(query)

            relevance_list = []
            for i in range(len(neighbor_indexes[0])):
                relevance_list.append(self.paper_ids[neighbor_indexes[0][i]])
            #top_words['Relevance'] = set(self.get_top_words(relevance_list, num_words=50))
            top_words['Relevance'] = set(relevance_list[:10])

            for aspect in self.aspects:
                aspect_list = self.re_rank(relevance_list, aspect)
                #top_words[aspect] = set(self.get_top_words(aspect_list, num_words=50))
                top_words[aspect] = set(aspect_list[:10])

            for aspect in top_words:
                for other_aspect in top_words:
                    denominator = len(top_words[aspect].union(top_words[other_aspect]))
                    enumerator = len(top_words[aspect].intersection(top_words[other_aspect]))
                    jaccard_dict[aspect, other_aspect].append(enumerator/denominator)

        for pair in jaccard_dict:
            print('{},{},{}'.format(pair[0], pair[1], np.mean(jaccard_dict[pair])))






def main():
    #data_dir = '/Users/saarkuzi/papers_to_index/'
    #get_titles(sys.argv[1])
    #robustness_evaluations('/Users/saarkuzi/Desktop/eval_acl.txt')
    #filter_queries('/home/skuzi2/iclr_large/scholar_queries.txt')
    query = ['knowledge graph', 'question answering', 'self attention']

    data_name = sys.argv[1]

    data_dir = '/home/skuzi2/{}_dataset/data_splits/dim.all.mod.neu.para.1.test.val'.format(data_name)
    aspects_dir = '/home/skuzi2/{}_dataset/{}_aspects.txt'.format(data_name, data_name)
    citations_dir = '/home/skuzi2/{}_dataset/citation_counts.txt'.format(data_name)
    titles_dir = '/home/skuzi2/{}_dataset/{}_titles.txt'.format(data_name, data_name)
    years_dir = '/home/skuzi2/{}_dataset/years.txt'.format(data_name, data_name)

    se = SearchEngine(data_dir + '.text.lemmatize', data_dir + '.ids', aspects_dir, citations_dir, titles_dir,
                      years_dir, filter_flag=False, years_flag='')
    #queries = [q.rstrip('\n') for q in open('/home/skuzi2/{}_dataset/phrase_queries.txt'.format(data_name), 'r').readlines()]
    queries = ['domain adaptation', 'matrix factorization']
    se.analyze_queries(queries)

    #se.run_jaccard(queries)
    #se.run_dataset('/home/skuzi2/{}_dataset/phrase_queries.txt'.format(data_name))





if __name__ == '__main__':
    main()
