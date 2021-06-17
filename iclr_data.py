import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm
import openreview
import sys

year = sys.argv[1]

def download_iclr19(client, outdir='./', get_pdfs=False):

    print('getting metadata...')
    submissions = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/20{}/Conference/-/Blind_Submission'.format(year))
    submissions_by_forum = {n.forum: n for n in submissions}
    print(len(submissions_by_forum))

    reviews = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/20{}/Conference/-/Paper.*/Official_Review'.format(year))
    reviews_by_forum = defaultdict(list)
    for review in reviews:
        reviews_by_forum[review.forum].append(review)

    meta_reviews = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/20{}/Conference/-/Paper.*/Meta_Review'.format(year))
    meta_reviews_by_forum = {n.forum: n for n in meta_reviews}

    metadata = []
    for forum in submissions_by_forum:

        forum_reviews = reviews_by_forum[forum]
        review_ratings = [n.content['rating'] for n in forum_reviews]

        #forum_meta_review = meta_reviews_by_forum[forum]
        #decision = forum_meta_review.content['recommendation']
        decision = 'nan'

        submission_content = submissions_by_forum[forum].content

        forum_metadata = {
            'forum': forum,
            'review_ratings': review_ratings,
            'decision': decision,
            'submission_content': submission_content
        }
        metadata.append(forum_metadata)

    print('writing metadata to file...')
    # write the metadata, one JSON per line:
    with open(os.path.join(outdir, 'iclr{}_metadata.jsonl'.format(year)), 'w') as file_handle:
        for forum_metadata in metadata:
            file_handle.write(json.dumps(forum_metadata) + '\n')

    # if requested, download pdfs to a subdirectory.
    if get_pdfs:
        pdf_outdir = os.path.join(outdir, 'iclr{}_pdfs'.format(year))
        os.makedirs(pdf_outdir)
        for forum_metadata in tqdm(metadata, desc='getting pdfs'):
            pdf_binary = client.get_pdf(forum_metadata['forum'])
            pdf_outfile = os.path.join(pdf_outdir, '{}.pdf'.format(forum_metadata['forum']))
            with open(pdf_outfile, 'wb') as file_handle:
                file_handle.write(pdf_binary)


if __name__ == '__main__':

    outdir = '../iclr{}'.format(year)
    base_url = 'https://api.openreview.net'
    if not os.path.exists(outdir): os.mkdir(outdir)

    client = openreview.Client(
        baseurl=base_url,
        username='saarkuzi@gmail.com',
        password='Kuz260487')


    download_iclr19(client, outdir, get_pdfs=True)