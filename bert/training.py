from bert_tokenizer import BertTokenizer
#import tensorflow as tf
#from tensorflow.keras import backend as K


data_dir = '/home/skuzi2/education_dataset/data_splits/dim.all.mod.neu.para.1.train.text'
bert_tokenizer = BertTokenizer()
with open(data_dir, 'r') as input_file:
    for i, line in enumerate(input_file):

        input_ids, input_mask, segment_ids = bert_tokenizer.convert_single_example(line, max_seq_length=100)
        print('----------------')
        print(line)
        print(input_ids)
        print(input_mask)
        print(segment_ids)