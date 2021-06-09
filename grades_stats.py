from collections import defaultdict, Counter
import numpy as np

base_dir = '/Users/saarkuzi/iclr17_dataset/'
grades_dir = base_dir + 'annotation_fixed.tsv'

train_ids = [i.rstrip('\n') for i in open(base_dir + 'train.ids').readlines()]
train_ids += [i.rstrip('\n') for i in open(base_dir + 'val.ids').readlines()]
test_ids = [i.rstrip('\n') for i in open(base_dir + 'test.ids').readlines()]
#train_ids = test_ids

grades_dict = defaultdict(list)
with open(grades_dir, 'r') as grades:
    for line in grades:
        args = line.rstrip('\n').split('\t')
        paper_id = args[0]
        grades = args[2:10]
        grades_dict[paper_id].append(grades)

std_dict = {dim: [] for dim in range(8)}
agreement_dict = {dim: [] for dim in range(8)}
dim_counters = {dim: 0 for dim in range(8)}
num_reviewers = {dim: [] for dim in range(8)}

for paper_id in train_ids:
    paper_grades = grades_dict[paper_id]

    for dim in range(8):
        scores = []
        for reviewer in paper_grades:
            if reviewer[dim] != '-' and reviewer[dim] != '' and reviewer[dim] != 's':
                scores.append(float(reviewer[dim]))

        if len(scores) > 0:
            dim_counters[dim] += 1
            scores_counter = dict(Counter(scores))
            high_i = sorted(scores_counter, key=scores_counter.get, reverse=True)[0]
            agreement_dict[dim].append(scores_counter[high_i])
            std_dict[dim].append(np.std(scores))
            num_reviewers[dim].append(len(scores))

for dim in std_dict:
    #print('{},{},{},{}'.format(dim, np.mean(std_dict[dim]), np.mean(agreement_dict[dim]), dim_counters[dim]))
    print('{},{},{},{},{}'.format(dim, np.mean(num_reviewers[dim]), np.std(num_reviewers[dim]),
                                  np.min(num_reviewers[dim]), np.max(num_reviewers[dim])))