import sys

ids_file = sys.argv[1]
fix_file = sys.argv[2]

ids = [i.rstrip('\n') for i in open(ids_file, 'r').readlines()]
remove = ['197', '287']

output = open(fix_file + '.fixed', 'w+')

with open(fix_file, 'r') as input_file:
    for i, line in enumerate(input_file):
        if ids[i] not in remove:
            output.write(line)