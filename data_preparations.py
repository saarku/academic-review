import os
import json


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




def main():
    input_folder_dir = '/Users/saarkuzi/iclr17_dataset/test'
    output_file_dir = '/Users/saarkuzi/iclr17_dataset/test'

    # transform_json_to_line_format(input_folder_dir, output_file_dir)

    aggregate_scores('/Users/saarkuzi/iclr17_dataset/annotation_fixed.tsv',
                      '/Users/saarkuzi/iclr17_dataset/annotation_aggregated.tsv')

    # check that the grades are correct

    aggregated_scores_v2()
if __name__ == '__main__':
    main()
