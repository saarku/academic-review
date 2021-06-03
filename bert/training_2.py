from model import BertModel

model = BertModel()
compiled = model.create_model()
print(compiled.summary)