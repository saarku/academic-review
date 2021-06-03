from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

train_texts = open('/home/skuzi2/education_dataset/data_splits/dim.all.mod.neu.para.1.train.text', 'r').readlines()

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
print(train_encodings)