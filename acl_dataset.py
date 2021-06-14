output_file = open('dim.all.mod.neu.para.1.test.val.text', 'w')
output_ids = open('dim.all.mod.neu.para.1.test.val.ids', 'w')
input_dir = '/Users/saarkuzi/papers_to_index/'
import os
import re
html_converter = re.compile(r'<[^>]+>')

for file_name in os.listdir(input_dir):
    file_id = file_name.split('.')[0]
    lines = [line.rstrip('\n') for line in open(input_dir + file_name, 'r').readlines()]
    output = ''
    for line in lines:
        output += html_converter.sub('', line) + ' '
    output_file.write(output + '\n')
    output_ids.write(file_id + '\n')