from model import BertModel


model = BertModel()
compiled = model.create_model(['bin'])
print(compiled.summary)