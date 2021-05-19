import sys
import os

para_ids = '/home/skuzi2/education_dataset/data_splits/dim.all.mod.neu.para.3.test.val.ids'
all_ids = '/home/skuzi2/education_dataset/data_splits/dim.all.mod.neu.para.1.test.val.ids'
para_ids = [i.rstrip('\n') for i in open(para_ids, 'r').readlines()]
all_ids = [i.rstrip('\n') for i in open(all_ids, 'r').readlines()]
ids = {'1': all_ids, '3': para_ids}

fix_folder = sys.argv[2]
os.mkdir(fix_folder + '_fixed')
remove = ['197', '287']


for file_name in os.listdir(fix_folder):
    args = file_name.split('.')
    if args[-1] == 'val':
        para = args[5]
        output = open(fix_folder + '_fixed/' + file_name, 'w+')
        with open(fix_folder + '/' + file_name, 'r') as input_file:
            for i, line in enumerate(input_file):
                if ids[para][i] not in remove:
                    output.write(line)
        output.close()