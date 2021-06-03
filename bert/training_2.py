import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel, BertTokenizerFast
from transformers import Trainer, TrainingArguments
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def read_20newsgroups(test_size=0.2):
    dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
    documents = dataset.data
    labels = dataset.target
    return train_test_split(documents, labels, test_size=test_size), dataset.target_names


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


model_name = 'allenai/scibert_scivocab_uncased'
max_length = 20

print('initialize tokenize')
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
#tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
(train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()

print('tokenizing')
train_encodings = tokenizer(train_texts[:10], truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts[:10], truncation=True, padding=True, max_length=max_length)

print('dataset')

train_dataset = NewsGroupsDataset(train_encodings, [float(i) for i in train_labels[:10]])
valid_dataset = NewsGroupsDataset(valid_encodings, [float(i) for i in valid_labels[:10]])

print('load')

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
print(model)
#model = AutoModel.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=200,               # log & save weights each logging_steps
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

print('train')
trainer.train()


test_encodings = tokenizer(valid_texts[:10], truncation=True, padding=True, max_length=max_length, return_tensors="pt")
outputs = model(**test_encodings, output_hidden_states=True)
print(outputs[1][0].shape)



