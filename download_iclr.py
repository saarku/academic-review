json_dir = '/Users/saarkuzi/Desktop/OpenReviewExplorer-master/data/iclr2017.json'
import json
import os
import openreview

base_dir = '/home/skuzi2/iclr_large/meta_data'
output_dir = '/home/skuzi2/iclr_large/pdfs'
if not os.path.exists(output_dir): os.mkdir(output_dir)

client = openreview.Client(
        baseurl='https://api.openreview.net',
        username='saarkuzi@gmail.com',
        password='Kuz260487')

for y in ['17', '18', '19', '20']:
    json_dir = base_dir + '/iclr20' + y + '.json'
    with open(json_dir, 'r') as data_file:
        json_data = data_file.read()
    data = json.loads(json_data)
    num_papers = len(data)

    for i, o in enumerate(data):
        if i%100 == 0: print('year:{} {}/{}'.format(y, i, num_papers))
        paper_id = o['url'].split('=')[1]
        try:
            pdf_binary = client.get_pdf(paper_id)
        except:
            pdf_binary = None
            
        if pdf_binary is not None:
            pdf_outfile = os.path.join(output_dir, '{}.pdf'.format(paper_id))
            with open(pdf_outfile, 'wb') as file_handle:
                file_handle.write(pdf_binary)