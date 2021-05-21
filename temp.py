'''
base_dir = '/Users/saarkuzi/iclr17_dataset'
test_ids = base_dir + '/test.ids'
val_ids = base_dir + '/val.ids'

val_ids = [i.rstrip('\n') for i in open(val_ids, 'r').readlines()]


text = '/Users/saarkuzi/Desktop/dim.all.mod.neu.para.1.test.val.text'
ids = '/Users/saarkuzi/Desktop/dim.all.mod.neu.para.1.test.val.ids'

texts = [i.rstrip('\n') for i in open(text, 'r').readlines()]
ids = [i.rstrip('\n') for i in open(ids, 'r').readlines()]

ids_to_add = []
text_to_add = []

for i in range(len(ids)):
    if ids[i] in val_ids:
        ids_to_add.append(ids[i])
        text_to_add.append(texts[i])


text = '/Users/saarkuzi/Desktop/dim.all.mod.neu.para.1.train.text'
ids = '/Users/saarkuzi/Desktop/dim.all.mod.neu.para.1.train.ids'

new_text_file = open(text + '.mod', 'w')
new_ids_file = open(ids + '.mod', 'w')

texts = [i.rstrip('\n') for i in open(text, 'r').readlines()] + text_to_add
ids = [i.rstrip('\n') for i in open(ids, 'r').readlines()] + ids_to_add



for i in range(len(ids)):
    new_ids_file.write(ids[i] + '\n')
    new_text_file.write(texts[i] + '\n')

'''








