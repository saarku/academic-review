#import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from transformers import AutoTokenizer


class BertTokenizer:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    def convert_single_example(self, text, max_seq_length=100):
        train_encodings = self.tokenizer([text], truncation=True, padding=True)
        print(train_encodings)
        '''
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
        '''
