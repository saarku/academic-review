import re
import sys

'<persName xmlns="http://www.tei-c.org/ns/1.0"><forename type="first">Nelson</forename><surname>Verastegui</surname></persName>'
'<orgName type="institution">H~res -France (++)[nstitut de Formation et ConseiL en Informatique</orgName>'
'<country key="CN">China</country>'
'<title level="m">Proceedings of the 4th Biennial International Workshop on Balto-Slavic Natural Language Processing</title>'


TAG_RE = re.compile(r'<[^>]+>')


def get_paper_fields(paper_dir):
    paper_lines = open(paper_dir, 'r').readlines()

    institutions = []
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
            institutions.append(institution)
    print(institutions)


get_paper_fields(sys.argv[1])