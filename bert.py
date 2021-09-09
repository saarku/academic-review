import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel, BertTokenizerFast
from transformers import Trainer, TrainingArguments
from feature_builder import FeatureBuilder
import numpy as np
import random
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


def load_data(data_name, dimension, data_type, num_samples=1000000):
    base_dir = '/home/skuzi2/{}_dataset/'.format(data_name)
    ids_dir = base_dir + 'data_splits/dim.all.mod.neu.para.1.{}.ids'.format(data_type)
    grades_dir = base_dir + 'annotations/annotation_aggregated.tsv'
    labels = FeatureBuilder.build_labels(ids_dir, grades_dir)
    lines = open(ids_dir.replace('ids', 'text'), 'r').readlines()
    x, y = FeatureBuilder.modify_data_to_dimension(lines, labels, dimension, num_samples=num_samples)
    y = [float(i) for i in y]
    return x, y


def fine_tune_bert(data_name, dimension, max_length, num_samples=1000):

    model_name = 'allenai/scibert_scivocab_uncased'

    print('Initializing Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    x_data, y_data = load_data(data_name, dimension, 'train', num_samples=num_samples)
    x_test, y_test = load_data(data_name, dimension, 'test.val')
    print('train size: {}, {}. test size: {}, {}'.format(len(x_data), len(y_data), len(x_test), len(y_test)))

    print('Tokenizing')
    train_encodings = tokenizer(x_data, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
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
    print('Finished')

    model_path = '../{}_dataset/bert_models_3/dim.{}.samples.{}'.format(data_name, dimension, num_samples)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print('Done')
    return model, tokenizer, train_encodings, test_encodings


def infer_embeddings(model, tokenizer, lines, output_dir, max_length):
    output_file = open(output_dir, 'w+')
    for i in range(0, len(lines), 16):
        print('infer step: {}'.format(i))
        end = min(i+16, len(lines))
        encodings = tokenizer(lines[i: end], truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        outputs = model(**encodings, output_hidden_states=True)
        hidden_states = outputs[1][-1].detach().numpy()  # (batch, seq, hidden)
        embeddings = np.mean(hidden_states, axis=1)  # (batch, hidden)
        for l in range(embeddings.shape[0]):
            line = ['({}, {})'.format(j, embeddings[l, j]) for j in range(embeddings.shape[1])]
            line = '[' + ', '.join(line) + ']\n'
            output_file.write(line)
    output_file.close()


def main():
    data_name = 'iclr17'#sys.argv[1]
    grade_dims = {'education': [0, 1, 2, 3, 4, 5, 6], 'iclr17': [1, 2, 3, 5, 6]}[data_name]
    max_length = 512
    samples = [50, 100, 150, 200, 250, 300, 350]

    for dim in grade_dims:
        for num_samples in samples:
            print('{}: {}, {}'.format(data_name, dim, num_samples))
            model, tokenizer, _, _ = fine_tune_bert(data_name, dim, max_length, num_samples=num_samples)

            print('inferring')
            for data_type in ['train', 'test.val']:
                output_dir = '../{}_dataset/bert_embeddings_3/dim.{}.samples.{}.{}'.format(data_name, dim, num_samples,
                                                                                         data_type)
                #encodings = train_encodings if data_type == 'train' else test_encodings
                data_dir = '/home/skuzi2/{}_dataset/data_splits/dim.all.mod.neu.para.1.{}.text'.format(data_name, data_type)
                lines = open(data_dir, 'r').readlines()
                infer_embeddings(model, tokenizer, lines, output_dir, max_length)


if __name__ == '__main__':
    main()
