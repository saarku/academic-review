import tensorflow_hub as hub

bert_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')
print(bert_model.summary())