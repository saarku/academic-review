import os
import json
from collections import defaultdict
import numpy as np


def transform_json_to_line_format(input_folder_dir, output_file_dir):
    """ Transfer a folder with json files to a single file with line per article.

    :param input_folder_dir: (string) the folder with the json files of the articles.
    :param output_file_dir: (string) the directory for the output line-based file.
    :return: None. Outputs the data to a file.
    """
    text_file = open(output_file_dir + '.text', 'w+')
    id_file = open(output_file_dir + '.ids', 'w+')
    for article_file_name in os.listdir(input_folder_dir):
        if article_file_name == '.DS_Store':
            continue
        article_id = article_file_name.split('.')[0]
        with open(input_folder_dir + '/' + article_file_name) as f:
            data = json.load(f)
            text = ''
            if data['metadata']['title'] is not None:
                text += data['metadata']['title'].lower()

            if data['metadata']['sections'] is not None:
                for section in data['metadata']['sections']:
                    text += ' ' + section['text'].replace('\n', ' ').lower()
            text_file.write(text + '\n')
            id_file.write(article_id + '\n')
    text_file.close()
    id_file.close()


def aggregate_scores(input_scores_dir, output_aggregated_dir):
    """ Aggregate the scores per articles over different reviews.

    :param input_scores_dir: (string) tab-separated file with scores of all reviewers.
    :param output_aggregated_dir: (string) the directory for the output aggregated scores.
    :return: None. Outputs the model to a file.
    """
    num_dimensions = 8
    all_grades_dict = {}
    aggregated_grades_dict = {}

    # Collect grades of all reviewers.
    for line in open(input_scores_dir, 'r').readlines()[1:]:
        arguments = line.rstrip('\n').split('\t')
        article_id = arguments[0]
        grades = arguments[2:-2]
        all_grades_dict[article_id] = all_grades_dict.get(article_id, []) + [grades]

    # Aggregate grades.
    for article_id in all_grades_dict:
        counters = {}
        aggregated_grades = {}
        for grades in all_grades_dict[article_id]:

            for grade_id, grade in enumerate(grades):
                try:
                    float(grade)
                except:
                    continue
                counters[grade_id] = counters.get(grade_id, 0.0) + 1.0
                aggregated_grades[grade_id] = aggregated_grades.get(grade_id, 0.0) + float(grade)

        for grade_id in aggregated_grades:
            aggregated_grades[grade_id] /= counters[grade_id]
        aggregated_grades_dict[article_id] = aggregated_grades

    # Output the aggregated grades to a file.
    with open(output_aggregated_dir, 'w+') as output_file:
        for article_id in aggregated_grades_dict:
            line = article_id
            for grade_id in range(num_dimensions):
                line += '\t' + str(aggregated_grades_dict[article_id].get(grade_id, '-'))
            output_file.write(line + '\n')


def aggregated_scores_v2():
    with open('/Users/saarkuzi/iclr17_dataset/train/reviews/424.json') as f:
        data = json.load(f)
        for i in range(len(data['reviews'])):
            if 'OTHER_KEYS' in data['reviews'][i]:
                print('_______' + data['reviews'][i]['OTHER_KEYS'] + '_______')
                print(data['reviews'][i].keys())
                for x in ['RECOMMENDATION_UNOFFICIAL','SUBSTANCE','APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY','IMPACT']:
                    if x in data['reviews'][i]:
                        print(x + ": " + str(data['reviews'][i][x]))


def split_to_paragraphs(full_data_dir):
    """ Split the full reviews into paragraphs.

    :param full_data_dir: (string) directory for the file with reviews.
    :return: None. Outputs the paragraphs to files.
    """

    ids = open(full_data_dir + 'dim.all.mod.neu.para.1.test.val.ids', 'r').readlines()
    texts = open(full_data_dir + 'dim.all.mod.neu.para.1.test.val.text', 'r').readlines()
    new_id_file = open(full_data_dir + 'dim.all.mod.neu.para.3.test.val.ids', 'w')
    new_text_file = open(full_data_dir + 'dim.all.mod.neu.para.3.test.val.text', 'w')

    for i in range(len(texts)):
        t = texts[i].rstrip('\n').split()
        num_words = len(t)//3
        new_text_file.write(' '.join(t[0:num_words]) + '\n')
        new_text_file.write(' '.join(t[num_words:2*num_words]) + '\n')
        new_text_file.write(' '.join(t[2*num_words:]) + '\n')
        new_id_file.write(ids[i]*3)

    new_id_file.close()
    new_text_file.close()


def write_split_to_file(assignments_list, data_dict, output_dir):
    ids_file = open(output_dir + '.ids', 'w')
    text_file = open(output_dir + '.text', 'w')
    counter = 0

    for tup in assignments_list:
        counter += 1
        for line in data_dict[tup[0]]:

            ids_file.write(tup[0] + '\n')
            text_file.write(line + '\n')
    ids_file.close()
    text_file.close()


def split_by_grade(input_data_dir, grades_dir, dimensions):

    data_lines = [line.rstrip('\n') for line in open(input_data_dir + 'dim.all.mod.neu.para.1.train.text', 'r').readlines()]
    data_ids = [line.rstrip('\n') for line in open(input_data_dir + 'dim.all.mod.neu.para.1.train.ids', 'r').readlines()]
    data_dict = defaultdict(list)

    for i in range(len(data_ids)):
        data_dict[data_ids[i]].append(data_lines[i])

    grades_dict = defaultdict(dict)
    grade_lines = [line.rstrip('\n').split('\t') for line in open(grades_dir, 'r').readlines()]

    for grade in grade_lines:
        if grade[0] not in data_dict: continue
        for i in range(1, len(grade)):
            if grade[i] != '-':
                grades_dict[grade[0]][i-1] = float(grade[i])

    for dim in dimensions:
        assignments = []
        for assignment_id in grades_dict:
            if dim in grades_dict[assignment_id]:
                assignments.append((assignment_id, grades_dict[assignment_id][dim]))

        assignments = sorted(assignments, key=lambda x: x[1])
        neg_assignments = assignments[:len(assignments)//2]
        pos_assignments = assignments[len(assignments)//2:]

        write_split_to_file(neg_assignments, data_dict, input_data_dir + 'dim.{}.mod.{}.para.3.train'.format(dim, 'neg'))
        write_split_to_file(pos_assignments, data_dict, input_data_dir + 'dim.{}.mod.{}.para.3.train'.format(dim, 'pos'))


def main():
    input_folder_dir = '/Users/saarkuzi/iclr17_dataset/train.paragraphs'
    output_file_dir = '/Users/saarkuzi/iclr17_dataset/test'
    grades_dir = '/Users/saarkuzi/iclr17_dataset/annotation_aggregated.tsv'

    # transform_json_to_line_format(input_folder_dir, output_file_dir)

    #aggregate_scores('/Users/saarkuzi/iclr17_dataset/annotation_fixed.tsv',
    #                  '/Users/saarkuzi/iclr17_dataset/annotation_aggregated.tsv')

    #split_to_paragraphs('/home/skuzi2/iclr17_new_dataset/data_splits/')

    split_by_grade('/home/skuzi2/iclr17_new_dataset/data_splits/',
                   '/home/skuzi2/iclr17_new_dataset/annotations/annotation_aggregated.tsv',
                   [1, 2, 3, 5, 6])


if __name__ == '__main__':
    main()
