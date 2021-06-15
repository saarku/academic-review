output_file = open('dim.all.mod.neu.para.1.test.val.text', 'w')
output_ids = open('dim.all.mod.neu.para.1.test.val.ids', 'w')
input_dir = '/Users/saarkuzi/papers_to_index/'
import os
import re
html_converter = re.compile(r'<[^>]+>')

'''
for file_name in os.listdir(input_dir):
    file_id = file_name.split('.')[0]
    lines = [line.rstrip('\n') for line in open(input_dir + file_name, 'r').readlines()]
    output = ''
    for line in lines:
        output += html_converter.sub('', line) + ' '
    output_file.write(output + '\n')
    output_ids.write(file_id + '\n')
'''

base_dir = '/home/skuzi2/iclr17_dataset/models/'

directories = {
    'Clarity': 'dim.1.algo.regression.uni.false.comb.comb_sum.model.lda.para.1_3.topics.cv.kl.kl.mode.neg_neu_pos.type.ovb.samedim.False.comb.comb_sum.predict.acl',
    'Originality': 'dim.2.algo.regression.uni.false.comb.comb_sum.model.lda.para.1_3.topics.cv.kl.kl.mode.neg_neu_pos.type.ovb.samedim.False.comb.comb_sum.predict.acl',
    'Soundness': 'dim.3.algo.regression.uni.false.comb.feature_comb.model.lda.para.1_3.topics.cv.kl.kl.mode.neg_neu_pos.type.ovb.samedim.True.predict.acl',
    'Sunstance': 'dim.5.algo.regression.uni.false.comb.comb_sum.model.lda.para.1_3.topics.cv.kl.kl.mode.neg_neu_pos.type.ovb.samedim.False.comb.comb_sum.predict.acl'
}

ids_dir = '/home/skuzi2/acl_dataset/data_splits/dim.all.mod.neu.para.1.test.val.ids'
ids = [i.rstrip('\n') for i in open(ids_dir, 'r').readlines()]

output_file = open('acl_aspects.txt', 'w')
dims = ['Clarity', 'Originality', 'Soundness', 'Sunstance']
output_file.write('id,' + ','.join(dims) + '\n')

for dim in dims:
    grades = [i.rstrip('\n') for i in open(base_dir + directories[dim], 'r').readlines()]
    directories[dim] = grades

for i in range(len(ids)):
    output_file.write('{},{},{},{},{}\n'.format(ids[i], directories['Clarity'][i], directories['Originality'][i],
                                              directories['Soundness'][i], directories['Sunstance'][i]))
output_file.close()