import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel, BertTokenizerFast
from transformers import Trainer, TrainingArguments
from feature_builder import FeatureBuilder
import numpy as np
import sys


class AcademicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def load_data(data_name, dimension, data_type):
    base_dir = '/home/skuzi2/{}_dataset/'.format(data_name)
    ids_dir = base_dir + 'data_splits/dim.all.mod.neu.para.1.{}.ids'.format(data_type)
    grades_dir = base_dir + 'annotations/annotation_aggregated.tsv'
    labels = FeatureBuilder.build_labels(ids_dir, grades_dir)
    lines = open(ids_dir.replace('ids', 'text'), 'r').readlines()
    x, y = FeatureBuilder.modify_data_to_dimension(lines, labels, dimension)
    y = [float(i) for i in y]
    return x, y


def fine_tune_bert(data_name, dimension, max_length):

    model_name = 'allenai/scibert_scivocab_uncased'

    print('Initializing Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    x_data, y_data = load_data(data_name, dimension, 'train')

    print('Tokenizing')
    train_encodings = tokenizer(x_data, truncation=True, padding=True, max_length=max_length)
    train_dataset = AcademicDataset(train_encodings, y_data)

    print('Loading BERT')
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
    )

    print('Fine-tuning')
    trainer.train()

    model_path = '../{}_dataset/bert_models/dim.{}'.format(data_name, dimension)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return model, tokenizer


def infer_embeddings(model, tokenizer, data_name, max_length, data_type, output_dir):
    base_dir = '/home/skuzi2/{}_dataset/'.format(data_name)
    directory = base_dir + 'data_splits/dim.all.mod.neu.para.1.' + data_type + '.text'
    lines = open(directory, 'r').readlines()
    encodings = tokenizer(lines, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    outputs = model(**encodings, output_hidden_states=True)
    hidden_states = outputs[1][-1].detach().numpy()  # (batch, seq, hidden)
    embeddings = np.mean(hidden_states, axis=1)  # (batch, hidden)
    output_file = open(output_dir, 'w+')

    for i in range(embeddings.shape[0]):
        line = ['({}, {})'.format(j, embeddings[i, j]) for j in range(embeddings.shape[1])]
        line = '[' + ', '.join(line) + ']\n'
        output_file.write(line)

    output_file.close()


def main():
    data_name = sys.argv[1]
    grade_dims = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]
    max_length = 512

    for dim in grade_dims:
        print('{}: {}'.format(data_name, dim))
        model, tokenizer = fine_tune_bert(data_name, dim, max_length)

        for data_type in ['train', 'test.val']:
            output_dir = '../{}_dataset/bert_embeddings/dim.{}.{}'.format(data_name, dim, data_type)
            infer_embeddings(model, tokenizer, data_name, max_length, data_type, output_dir)


if __name__ == '__main__':
    main()
