output_file = open('table.txt', 'w+')

with open('temp.txt', 'r') as input_file:
    for line in input_file:
        values = line.rstrip('\n').split('\t')
        str_values = []
        for v in values:
            str_v = str(v).split('.')
            if str_v[0][0] == '-':
                str_values.append('$-.' + str_v[1] + '$')
            else:
                str_values.append('$.' + str_v[1] + '$')
        output_file.write(' & '.join(str_values) + '\n')
output_file.close()

'''
with open('temp.txt', 'r') as input_file:
    for line in input_file:
        values = line.rstrip('\n').split('\t')
        str_values = []
        for v in values:
            str_values.append(v)
        output_file.write(' & '.join(str_values) + '\n')
output_file.close()
'''

