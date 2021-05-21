import random
base_dir = '/Users/saarkuzi/iclr17_dataset/'


grades = {}
with open(base_dir + 'annotation_aggregated.tsv', 'r') as input_file:
    for line in input_file:
        args = line.rstrip('\n').split('\t')
        grades[args[0]] = args[1:]

paper_ids = []

for paper in grades:
    if grades[paper][1] != '-' and grades[paper][2] != '-' and grades[paper][3]:
        paper_ids.append(paper)

texts = []
ids = []

for data_type in ['train', 'test', 'val']:
    ids += [line.rstrip('\n') for line in open(base_dir + data_type + '.ids', 'r').readlines()]
    texts += [line.rstrip('\n') for line in open(base_dir + data_type + '.ids', 'r').readlines()]

texts = list(zip(ids, texts))
distilled = [element for element in texts if element[0] in paper_ids]
texts = distilled
random.shuffle(texts)

test_size = int(len(texts) * 0.2)

test_file = open('dim.all.mod.neu.para.1.test.text', 'w+')
test_id_file = open('dim.all.mod.neu.para.1.test.ids', 'w+')

for i in range(test_size):
    test_file.write(texts[i][1] + '\n')
    test_id_file.write(texts[i][0] + '\n')


train_file = open('dim.all.mod.neu.para.1.train.val.text', 'w+')
train_id_file = open('dim.all.mod.neu.para.1.train.val.ids', 'w+')

for i in range(test_size, len(texts)):
    train_file.write(texts[i][1] + '\n')
    train_id_file.write(texts[i][0] + '\n')