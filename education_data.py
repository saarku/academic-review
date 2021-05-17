import re
base_dir = '/Users/saarkuzi/Documents/PycharmProjects/education/topic_grading/datasets/'

grades_file = open('annotation_aggregated.tsv', 'w')
output_train = open('dim.all.mod.neu.para.1.train.text', 'w')
output_test = open('dim.all.mod.neu.para.1.test.text', 'w')
output_train_ids = open('dim.all.mod.neu.para.1.train.ids', 'w')
output_test_ids = open('dim.all.mod.neu.para.1.test.ids', 'w')

counter = 0

train_lines = open(base_dir + '33072-17/alltext/alltext.dat', 'r').readlines()
train_grades = open(base_dir + '33072-17/metadata.dat', 'r').readlines()

for i in range(len(train_lines)):
    output_train.write(train_lines[i])
    output_train_ids.write(str(counter) + '\n')
    grades_file.write(str(counter) + '\t' + '\t'.join(train_grades[i].split('\t')[1:]))
    counter += 1

test_lines = open(base_dir + '33072-18/alltext/alltext.dat', 'r').readlines()
test_grades = open(base_dir + '33072-18/metadata.dat', 'r').readlines()

for i in range(len(test_lines)):
    html_converter = re.compile(r'<[^>]+>')
    line = html_converter.sub('', test_lines[i])

    output_test.write(line)
    output_test_ids.write(str(counter) + '\n')
    grades_file.write(str(counter) + '\t' + '\t'.join(test_grades[i].split('\t')[1:]))
    counter += 1