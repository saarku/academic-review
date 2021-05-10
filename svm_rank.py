import os
import time
import numpy as np


class SVMRank:

    def __init__(self):
        self.svm_dir = './svm_rank/'
        self.model_dir = None

    def fit(self, training_data, training_labels, model_dir, c, kernel=0):
        training_dir = self.create_svm_data(training_data, labels=training_labels)
        command = self.svm_dir + 'svm_rank_learn -c ' + str(c) + ' -t ' + str(kernel) + ' ' + training_dir + ' '
        command += model_dir
        os.system(command)
        os.system('rm ' + training_dir)
        self.model_dir = model_dir

    def predict(self, test_data):
        test_dir = self.create_svm_data(test_data)
        predictions_dir = str(time.time())
        command = self.svm_dir + 'svm_rank_classify ' + test_dir + ' ' + self.model_dir + ' ' + predictions_dir
        os.system(command)
        predictions = [i.rstrip('\n').split('\t')[-1] for i in open(predictions_dir, 'r').readlines()]
        os.system('rm ' + predictions_dir)
        return np.reshape(np.asarray(predictions), (-1, 1))

    @staticmethod
    def create_svm_data(data, labels=None):
        output_dir = str(time.time())
        output_file = open(output_dir, 'w')
        for i in range(data.shape[0]):
            line = str(labels[i]) if labels is not None else '1'
            line += ' qid:1 '
            for j in range(data.shape[1]):
                line += str(j+1) + ':' + str(data[i, j]) + ' '
            output_file.write(line + '\n')
        output_file.close()
        return output_dir