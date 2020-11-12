import json
import os

ref_comments = {}

for file_dir in os.listdir('/Users/saarkuzi/iclr17_dataset/reviews'):
    if file_dir == '.DS_Store': continue
    ref_id = file_dir.rstrip('.json')
    with open('/Users/saarkuzi/iclr17_dataset/reviews/'+ref_id+'.json') as f:
        data = json.load(f)
        for i in range(len(data['reviews'])):
            if 'OTHER_KEYS' in data['reviews'][i]:
                if len(data['reviews'][i]['comments']) > 1:
                    c = data['reviews'][i]['comments']
                    c = c.lower().replace('\n', ' ').split()
                    ref_comments[ref_id] = ref_comments.get(ref_id, []) + [c]


used_ref_id = []
lines = [line.rstrip('\n') for line in open('/Users/saarkuzi/iclr17_dataset/annotation_full.tsv', 'r').readlines()]
mapping = {}
for line in lines[1:len(lines)]:
    args = line.split('\t')
    comment_id = args[0]
    comment = args[10].lower().replace('\n', ' ').split()

    for ref_id in ref_comments:
        for c in ref_comments[ref_id]:
            if '_'.join(c) == '_'.join(comment):
                mapping[comment_id] = ref_id
                used_ref_id.append(ref_id)
print(len(mapping))

for ref_id in ref_comments:
    if ref_id not in used_ref_id:
        print('missing: ' + ref_id)



with open('/Users/saarkuzi/iclr17_dataset/annotation_fixed.tsv', 'w+') as output_file:
    for line in lines[0:len(lines)]:
        args = line.split('\t')
        comment_id = args[0]
        ref_id = mapping.get(comment_id, '-1')

        args[0] = ref_id
        output_file.write('\t'.join(args) + '\n')