import tensorflow_hub as hub
from transformers import AutoModel, BertForSequenceClassification, Trainer, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
sentences = open('/home/skuzi2/education_dataset/data_splits/dim.all.mod.neu.para.1.train.text').readlines()
inputs = tokenizer(sentences, padding="max_length", truncation=True)
model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=1)

trainer = Trainer(model=model, train_dataset=inputs)
trainer.train()




'''
#model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')


#bert_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')
#print(bert_model.summary())

#'/home/skuzi2/scibert_scivocab_uncased''allenai/scibert_scivocab_uncased'

model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=1)
print(model)



trainer = Trainer(
    model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
)
'''