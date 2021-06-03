from bert_tokenizer import BertTokenizer
from model import BertModel
from bert_layer import BertLayer


#t = BertTokenizer()
#t.convert_single_example('public scope science depth')
#model = BertModel()

#bert_layer = BertLayer(n_fine_tune_layers=1, pooling="first")

model = BertModel()
model.create_model(['bin'])

print(model.summary())
