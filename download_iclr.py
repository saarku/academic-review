json_dir = '/Users/saarkuzi/Desktop/OpenReviewExplorer-master/data/iclr2017.json'
import json
import os
import openreview

base_dir = '/home/skuzi2/iclrlarge_dataset/meta_data'
#output_dir = '/home/skuzi2/iclr_large/pdfs'
download_flag = False
citation_flag = False
accept_flag = True
years_flag = False
#if not os.path.exists(output_dir): os.mkdir(output_dir)

client = openreview.Client(
        baseurl='https://api.openreview.net',
        username='saarkuzi@gmail.com',
        password='Kuz260487')

if years_flag:
    citation_counts = open('years.txt', 'w')
    for y in ['17', '18', '19', '20']:
        json_dir = base_dir + '/iclr20' + y + '.json'
        with open(json_dir, 'r') as data_file:
            json_data = data_file.read()
        data = json.loads(json_data)

        for i, o in enumerate(data):
            paper_id = o['url'].split('=')[1]
            citation_counts.write(paper_id + ',' + y + '\n')

if citation_flag:
    citation_counts = open('citation_counts.txt', 'w')
    for y in ['17', '18', '19', '20']:
        json_dir = base_dir + '/iclr20' + y + '.json'
        with open(json_dir, 'r') as data_file:
            json_data = data_file.read()
        data = json.loads(json_data)
        num_papers = len(data)

        for i, o in enumerate(data):
            if 'citations' in o:
                print(y)
                citations = str(int(o['citations']))
                paper_id = o['url'].split('=')[1]
                citation_counts.write(paper_id + ',' + citations + '\n')

if accept_flag:
    citation_counts = open('accept_decisions.txt', 'w')
    for y in ['17', '18', '19', '20']:
        json_dir = base_dir + '/iclr20' + y + '.json'
        with open(json_dir, 'r') as data_file:
            json_data = data_file.read()
        data = json.loads(json_data)
        num_papers = len(data)

        for i, o in enumerate(data):
            if 'decision' in o:
                decision = str(int(o['decision']))
                paper_id = o['url'].split('=')[1]
                if 'Accept' in decision:
                    citation_counts.write(paper_id + ',1\n')
                else:
                    citation_counts.write(paper_id + ',0\n')

if download_flag:
    output_dir = './'
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

