import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm
import openreview


def download_iclr19(client, outdir='./', get_pdfs=False):

    print('getting metadata...')
    # get all ICLR '19 submissions, reviews, and meta reviews, and organize them by forum ID
    # (a unique identifier for each paper; as in "discussion forum").
    submissions = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/2019/Conference/-/Blind_Submission')
    submissions_by_forum = {n.forum: n for n in submissions}

    # There should be 3 reviews per forum.
    reviews = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/2019/Conference/-/Paper.*/Official_Review')
    reviews_by_forum = defaultdict(list)
    for review in reviews:
        reviews_by_forum[review.forum].append(review)

    # Because of the way the Program Chairs chose to run ICLR '19, there are no "decision notes";
    # instead, decisions are taken directly from Meta Reviews.
    meta_reviews = openreview.tools.iterget_notes(
        client, invitation='ICLR.cc/2019/Conference/-/Paper.*/Meta_Review')
    meta_reviews_by_forum = {n.forum: n for n in meta_reviews}

    # Build a list of metadata.
    # For every paper (forum), get the review ratings, the decision, and the paper's content.
    metadata = []
    for forum in submissions_by_forum:

        forum_reviews = reviews_by_forum[forum]
        review_ratings = [n.content['rating'] for n in forum_reviews]

        forum_meta_review = meta_reviews_by_forum[forum]
        decision = forum_meta_review.content['recommendation']

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
    with open(os.path.join(outdir, 'iclr19_metadata.jsonl'), 'w') as file_handle:
        for forum_metadata in metadata:
            file_handle.write(json.dumps(forum_metadata) + '\n')

    # if requested, download pdfs to a subdirectory.
    if get_pdfs:
        pdf_outdir = os.path.join(outdir, 'iclr19_pdfs')
        os.makedirs(pdf_outdir)
        for forum_metadata in tqdm(metadata, desc='getting pdfs'):
            pdf_binary = client.get_pdf(forum_metadata['forum'])
            pdf_outfile = os.path.join(pdf_outdir, '{}.pdf'.format(forum_metadata['forum']))
            with open(pdf_outfile, 'wb') as file_handle:
                file_handle.write(pdf_binary)


if __name__ == '__main__':
    outdir = '../iclr2019'
    base_url = 'https://api.openreview.net'
    if not os.path.exists(outdir): os.mkdir(outdir)

    client = openreview.Client(
        baseurl=base_url,
        username='saarkuzi@gmail.com',
        password='Kuz260487')

    download_iclr19(client, outdir, get_pdfs=True)