import re
import os
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

    institutions = set()
    for line in paper_lines:
        if '<orgName type=\"institution\">' in line:
            institution = TAG_RE.sub('', line.rstrip('\n'))
            institution = institution.replace('\t', '')
            words = institution.split()
            processed = []
            for w in words:
                w = w.replace('-', '')
                w = w.replace('~', '')
                predicate = True
                for s in ['(', ')']:
                    if s in w: predicate = False
                if predicate: processed.append(w)
            institution = ' '.join(processed)
            institutions.add(institution)
    return institutions


def get_paper_authors(paper_dir):
    paper_lines = open(paper_dir, 'r').readlines()
    persons = []

    for line in paper_lines:
        if '<abstract>' in line: break
        if '<persName' in line:
            person = TAG_RE.sub(' ', line.rstrip('\n'))
            person = person.replace('\t', ' ')
            names = person.split()
            person = ' '.join(names)
            persons.append(person)
    return persons


def get_paper_conference(paper_dir):
    paper_lines = open(paper_dir, 'r').readlines()

    conference = ''
    for line in paper_lines:
        if '<title level=\"m\">' in line:
            conference = TAG_RE.sub('', line.rstrip('\n'))
            conference = conference.replace('\t', '')
            words = conference.split()
            processed = []
            for w in words:
                w = w.replace('-', '')
                w = w.replace('~', '')
                predicate = True
                for s in ['(', ')']:
                    if s in w: predicate = False
                if predicate: processed.append(w)
            conference = ' '.join(processed)
            break
    return conference


def main():
    input_dir = '/home/skuzi2/acl_antology_all_papers/grobid_output/'
    files = {'conferences': open('conferences', 'w'), 'authors': open('authors.txt', 'w'),
             'affiliations': open('affiliations.txt', 'w')}

    for file_name in os.listdir(input_dir):
        paper_dir = input_dir + file_name
        paper_id = file_name.split('.')[0]
        affiliations = get_paper_affiliation(paper_dir)
        authors = get_paper_authors(paper_dir)
        conference = get_paper_conference(paper_dir)
        files['conferences'].write(paper_id + ',' + conference + '\n')
        files['conferences'].flush()
        files['authors'].write(paper_id + ',' + ','.join(authors) + '\n')
        files['authors'].flush()
        files['affiliations'].write(paper_id + ',' + ','.join(affiliations) + '\n')
        files['affiliations'].flush()

    
if __name__ == '__main__':
    main()