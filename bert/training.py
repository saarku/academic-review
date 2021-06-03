from transformers import AutoTokenizer, AutoModel
import tensorflow as tf
from transformers import TFTrainer, TFTrainingArguments, TFDistilBertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
#model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

train_texts = open('/home/skuzi2/education_dataset/data_splits/dim.all.mod.neu.para.1.train.text', 'r').readlines()
train_labels = [1]*len(train_texts)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))




training_args = TFTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

print(model.summary())
'''
trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
)

trainer.train()
'''