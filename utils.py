from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import stem
from scipy.sparse import csr_matrix


def to_sparse(vectors, shape):
    """ Initializing a sparse matrix with a set of vectors.

    :param vectors: (list) the list of vectors.
    :param shape: (tuple) the required shape of the matrix.
    :return: csr_matrix.
    """
    rows, cols, data = [], [], []
    for i in range(len(vectors)):
        for tup in vectors[i]:
            col, point = tup[0], tup[1]
            rows += [i]
            cols += [col]
            data += [point]
    return csr_matrix((data, (rows, cols)), shape=shape)


def pre_process_text(text):
    """ Pre-process text including tokenization, stemming, and stopwords removal.

    :param text: (string) the input text for processing.
    :return: String. The processed text.
    """
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    word_tokens = [w for w in word_tokens if not w in stop_words]
    stemmer = stem.PorterStemmer()
    stemmed_text = [stemmer.stem(w) for w in word_tokens]
    filtered_text = [w for w in stemmed_text if not w in stop_words]
    text_len = min(1000, len(filtered_text))
    return " ".join(filtered_text[0:text_len])


def from_sparse(sparse_matrix, filter_list=None):
    """ Extract vectors from a sparse matrix.

    :param sparse_matrix: (csr_matrix).
    :param filter_list: (list).
    :return: String. The processed text.
    """
    data, indices, indptr = sparse_matrix.data, sparse_matrix.indices, sparse_matrix.indptr
    if filter_list is None:
        filter_list = []
    mat = []
    for row_id in range(len(indptr)-1):
        vec = []
        for i in range(indptr[row_id], indptr[row_id+1]):
            col_id = indices[i]
            point = data[i]
            if point not in filter_list:
                vec += [(col_id, point)]
        mat += [vec]
    return mat

