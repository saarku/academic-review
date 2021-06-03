import tensorflow_hub as hub
from transformers import AutoModel, TFBertForSequenceClassification

#model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')


#bert_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')
#print(bert_model.summary())

#'/home/skuzi2/scibert_scivocab_uncased'

model = TFBertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased')


