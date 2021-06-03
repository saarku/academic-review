import tensorflow_hub as hub
from bert.tokenization import FullTokenizer


class BertTokenizer:

    def __init__(self, sess):
        #self.bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        self.bert_path  = "/home/skuzi2/scibert_scivocab_uncased"
        self.sess = sess
        self.tokenizer = self.create_tokenizer_from_hub_module()

    def create_tokenizer_from_hub_module(self):
        #bert_module = hub.Module(self.bert_path)
        #tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        #vocab_file, do_lower_case = self.sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"], ])
        vocab_file = self.bert_path + '/vocab.txt'
        do_lower_case = True
        return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case, preserve_unused_tokens=False)

    def convert_single_example(self, text, max_seq_length=100):
        text_tokens = self.tokenizer.tokenize(text)
        if len(text_tokens) > max_seq_length - 2:
            text_tokens = text_tokens[:(max_seq_length - 2)]

        tokens, segment_ids = [], []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in text_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return input_ids, input_mask, segment_ids
