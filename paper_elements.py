import re
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import stem

'<persName xmlns="http://www.tei-c.org/ns/1.0"><forename type="first">Nelson</forename><surname>Verastegui</surname></persName>'
'<orgName type="institution">H~res -France (++)[nstitut de Formation et ConseiL en Informatique</orgName>'
'<country key="CN">China</country>'
'<title level="m">Proceedings of the 4th Biennial International Workshop on Balto-Slavic Natural Language Processing</title>'


TAG_RE = re.compile(r'<[^>]+>')


def pre_process_nltk(text, additional_stopwords=None):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    if additional_stopwords is not None:
        stop_words = stop_words.union(set(additional_stopwords))

    word_tokens = word_tokenize(text)
    word_tokens = [w for w in word_tokens if not w in stop_words]
    stemmer = stem.PorterStemmer()
    stemmed_text = [stemmer.stem(w) for w in word_tokens]
    filtered_text = [w for w in stemmed_text if not w in stop_words]
    return " ".join(filtered_text)


def get_paper_affiliation(paper_dir):
    paper_lines = open(paper_dir, 'r').readlines()

    institutions = []
    stemmed_institutions = []
    for line in paper_lines:
        if '<orgName type=\"institution\">' in line:
            institution = TAG_RE.sub('', line.rstrip('\n'))
            institution = institution.replace('\t', '')
            words = institution.split()
            processed = []
            for w in words:
                w = w.replace('-', '')
                predicate = True
                for s in ['(', ')']:
                    if s in w: predicate = False
                if predicate: processed.append(w)
            institution = ' '.join(processed)
            stemmed_institution = pre_process_nltk(institution)
            institutions.append(institution)
            stemmed_institutions.append(stemmed_institution)
    print(institutions)
    print(stemmed_institutions)


def get_paper_authors(paper_dir):
    paper_lines = open(paper_dir, 'r').readlines()
    names = []

    for line in paper_lines:
        if '<persName' in line:
            person = TAG_RE.sub(' ', line.rstrip('\n'))
            person = person.replace('\t', ' ')
            names = person.split()
            person = ' '.join(names)
            names.append(person)
    print(names)




get_paper_authors(sys.argv[1])