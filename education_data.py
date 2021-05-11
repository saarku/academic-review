base_dir = '/Users/saarkuzi/Documents/PhDResearch/EducationProject/TopicModels/data/'

grades_file = open('annotation_aggregated.tsv', 'w')
output_train = open('dim.all.mod.neu.para.1.train.text', 'w')
output_test = open('dim.all.mod.neu.para.1.test.ids', 'w')
output_train_ids = open('dim.all.mod.neu.para.1.train.ids', 'w')
output_test_ids = open('dim.all.mod.neu.para.1.test.ids', 'w')

counter = 0

train_lines = open(base_dir + '33072_1/v1.dat', 'r').readlines()
train_grades = open(base_dir + '33072_1/metadata.dat.old', 'r').readlines()

for i in range(len(train_lines)):
    output_train.write(train_lines[i])
    output_train_ids.write(str(counter) + '\n')
    grades_file.write(str(counter) + '\t' + train_grades[i])
    counter += 1

test_lines = open(base_dir + '33288/v1.dat', 'r').readlines()
test_grades = open(base_dir + '33288/metadata.dat.old', 'r').readlines()

for i in range(len(test_lines)):
    output_test.write(test_lines[i])
    output_test_ids.write(str(counter) + '\n')
    grades_file.write(str(counter) + '\t' + test_grades[i])
    counter += 1