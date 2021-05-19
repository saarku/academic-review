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


def pre_process(raw_data_dir, output_folder):
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    data_lines = [pre_process_text(line).split() for line in open(raw_data_dir, 'r').read().split('\n')][0:-1]
    print(data_lines)

    with open(output_folder + '/raw_data.txt', 'w+') as raw_file:
        for words in data_lines:
            words_encoded = [w for w in words]
            raw_file.write(' '.join(words_encoded) + '\n')

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

    with open(output_folder + '/full_data.txt', 'w+') as f:
        for tokens in x_token_ids:
            for token in tokens:
                f.write(str(token) + ' ')
            f.write('\n')


def main():
    raw_data_dir = sys.argv[1]
    output_folder = sys.argv[2]
    pre_process(raw_data_dir, output_folder)


if __name__ == '__main__':
    main()
