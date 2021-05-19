import itertools
import numpy as np
import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import stem


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


def pre_process(train_data_dir, test_data_dir, output_folder):
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    train_lines = [pre_process_text(line).split() for line in open(train_data_dir, 'r').read().split('\n')][0:-1]
    test_lines = [pre_process_text(line).split() for line in open(test_data_dir, 'r').read().split('\n')][0:-1]
    data_lines = train_lines + test_lines

    all_tokens = itertools.chain.from_iterable(data_lines)
    word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
    all_tokens = itertools.chain.from_iterable(data_lines)
    id_to_word = [token for idx, token in enumerate(set(all_tokens))]
    id_to_word = np.asarray(id_to_word)

    x_token_ids = [[word_to_id[token] for token in x] for x in data_lines]
    count = np.zeros(id_to_word.shape)
    for x in x_token_ids:
        for token in x:
            count[token] += 1
    indices = np.argsort(-count)
    id_to_word = id_to_word[indices]

    word_to_id = {token: idx for idx, token in enumerate(id_to_word)}
    x_token_ids = [[word_to_id.get(token, -1) + 1 for token in x] for x in data_lines]

    np.save(output_folder + '/dictionary.npy', np.asarray(id_to_word))

    train_file = open(output_folder + '/train.txt', 'w+')
    test_file = open(output_folder + '/test.txt', 'w+')
    num_train = len(train_lines)

    for i, tokens in enumerate(x_token_ids):
        output_file = train_file if i < num_train else test_file
        for token in tokens:
            output_file.write(str(token) + ' ')
        output_file.write('\n')

    train_file.close()
    test_file.close()


def main():
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    output_folder = sys.argv[3]
    pre_process(train_dir, test_dir, output_folder)


if __name__ == '__main__':
    main()
