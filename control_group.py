__author__ = 'hsuanwei'

from os import walk
from sys import argv
from math import log
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import os
import re
import logging
import timeit
import multiprocessing


def get_data(raw_data_file):
    label_data = []
    feature_data = []
    for each_line in open(raw_data_file).readlines():
        split_line = each_line.rstrip('\n').split('\t')
        for each_column in range(1, len(split_line)):
            split_line[each_column] = split_line[each_column].split(':')[1]
        label_data.append(split_line[0])
        feature_data.append(split_line[1:])
    label_data = np.array(label_data)
    feature_data = np.array(feature_data)
    return label_data, feature_data


def get_feature_idx(feature_idx_directory):
    file_list = []
    feature_index = {}
    for (dir_path, dir_names, file_names) in walk(feature_idx_directory):
        file_list.extend(file_names)
    for each_file in file_list:
        feature_index_data = np.loadtxt(feature_idx_directory + '/' + each_file)
        feature_list = feature_index_data[:, 0].astype(int).tolist()
        feature_index[each_file.split('_')[0]] = {'original': feature_list, 'selected': []}

    return feature_index


def train_test_split(label_data, feature_data):
    train_test_split_group = []
    random_split = ShuffleSplit(n_splits=5, random_state=0, train_size=0.8, test_size=0.2)
    fold_count = 0
    for train_index, test_index in random_split.split(feature_data):
        fold_count += 1
        tmp_fold = [label_data[train_index], feature_data[train_index], label_data[test_index],
                    feature_data[test_index], fold_count]
        train_test_split_group.append(tmp_fold)
    return train_test_split_group


def svm_performance_evaluation(result_file, true_label):
    result_data = open(result_file, mode='r').readlines()
    pattern = re.compile('^(.*?)\s')
    predict_label = []
    for line_count in range(1, len(result_data)):
        predict_label.append(pattern.search(result_data[line_count]).group(1))
    precision, recall, fscore, support = precision_recall_fscore_support(true_label, predict_label, average='weighted')
    accuracy = accuracy_score(true_label, predict_label)
    return accuracy, precision, recall, fscore


def generate_feature_priority(label_data, feature_data, input_file_name, fold):

    def generate_tmp_file(label, feature, file_name='train_fold-0'):
        train_fold_file = open(file_name, mode='w')
        feature_amount = len(feature[0])
        tmp_output = ''
        for i in range(len(label)):
            tmp_feature = ''
            for n in range(feature_amount):
                tmp_feature = '{0}\t{1}:{2}'.format(tmp_feature, n + 1, feature[i][n])
            tmp_output = '{0}{1}{2}\n'.format(tmp_output, label[i], tmp_feature)
        train_fold_file.write(tmp_output)
        train_fold_file.close()
    output_file_name = '{0}_fold-{1}.train'.format(input_file_name, fold)
    output_dir = 'fs_fold-{0}'.format(fold)
    generate_tmp_file(label_data, feature_data, output_file_name)
    os.system('python feature_selection.py {0} -o {1} -un -a -sup -a'.format(output_file_name, output_dir))
    return output_dir


def cv_work(train_test_group):
    feature_index_dir = generate_feature_priority(train_test_group[0], train_test_group[1], train_test_group[6],
                                                  train_test_group[4])
    feature_index = get_feature_idx(feature_index_dir)
    output_result = {}
    method_list = feature_index.keys()
    for each_method in feature_index:
        logging.info('Fold : {0}\tMethod : {1}'.format(train_test_group[4], each_method))
        select = IntegrateFeatureSelection(feature_index[each_method]['original'], train_test_group[0], train_test_group[1],
                                           train_test_group[5], train_test_group[4])
        if train_test_group[7] == 0:
            fs_function = select.sffs_single_list
        elif train_test_group[7] == 1:
            fs_function = select.modified_sffs_single_list
        result_accuracy, selected_feature_set, iterations, time_required = fs_function(len(train_test_group[1][0]))
        indep_test = IndependentPredictor(train_test_group[0], train_test_group[1], train_test_group[2], train_test_group[3],
                                          selected_feature_set, train_test_group[4])
        indep_acc, indep_prc, indep_rc, indep_fs = indep_test.select_predictor(train_test_group[5])
        output_result[each_method] = 'Cross Validation : {0}\nProcessing Time:{1}sec\nIteration Count:{2}\nSelected Features:{3}\n' \
                     'Train Accuracy:{4}%\nTest Accuracy:{5}%\nTest Precision:{6}%\nTest Recall:{7}%\nTest F-score:{8}\n\n' \
            .format(train_test_group[4], time_required, iterations, selected_feature_set, result_accuracy,
                    100*indep_acc, 100*indep_prc, 100*indep_rc, indep_fs)
    logging.info('Done : {0}-Fold'.format(train_test_group[4]))
    return [output_result, method_list]


class IndependentPredictor:
    def __init__(self, train_label, train_feature, test_label, test_feature, feature_list, fold=0):
        self.train_label, self.train_feature = train_label, train_feature
        self.test_label, self.test_feature = test_label, test_feature
        self.feature_list = feature_list
        self.fold = fold

    def select_predictor(self, classifier):
        if classifier == 'svm':
            return self.svm()

    def svm(self):

        def generate_tmp_file(label_data, feature_data, feature_list, tmp_file_name='tmp_svm_file'):
            tmp_svm_file = open(tmp_file_name, mode='w')
            new_output = ''
            for line_count in range(len(label_data)):
                current_line = feature_data[line_count]
                new_line = label_data[line_count]
                for feature_count in range(len(feature_list)):
                    new_line += '\t{0}:{1}'.format(feature_count + 1, current_line[feature_list[feature_count]-1])
                new_output += '{0}\n'.format(new_line)
            tmp_svm_file.write(new_output)
            tmp_svm_file.close()

        def get_svm_grid_parameter(feature_dimensions, scale_file):
            default_gamma = log(1 / (float(feature_dimensions))) / log(2)
            gamma_start = default_gamma - 1
            gamma_stop = default_gamma + 1
            grid_result = os.popen('svm-grid -log2c -1,1,0.5 -log2g {0},{1},0.5 {2}'
                                   .format(gamma_start, gamma_stop, scale_file)).readlines()[-1].rstrip('\n').split(' ')
            return grid_result[0], grid_result[1]

        generate_tmp_file(self.train_label, self.train_feature, self.feature_list,
                          'tmp_svm_train-{0}'.format(self.fold))
        generate_tmp_file(self.test_label, self.test_feature, self.feature_list, 'tmp_svm_test-{0}'.format(self.fold))
        os.system('svm-scale -s scale-info-{0} tmp_svm_train-{0} > scale-tmp_svm_train-{0}'.format(self.fold))
        os.system('svm-scale -r scale-info-{0} tmp_svm_test-{0} > scale-tmp_svm_test-{0}'.format(self.fold))
        c_para, g_para = get_svm_grid_parameter(len(self.feature_list), 'scale-tmp_svm_train-{0}'.format(self.fold))
        os.popen(
            'svm-train -b 1 -h 0 -c {0} -g {1} scale-tmp_svm_train-{2} svm-model-{2}'.format(c_para, g_para, self.fold))
        os.popen('svm-predict -b 1 scale-tmp_svm_test-{0} svm-model-{0} predict_svm-{0}'.format(self.fold))
        accuracy, precision, recall, fscore = svm_performance_evaluation('predict_svm-{0}'.format(self.fold),
                                                                         self.test_label)
        tmp_file_list = ['tmp_svm_train', 'tmp_svm_test', 'scale-info', 'scale-tmp_svm_train',
                         'scale-tmp_svm_test', 'svm-model', 'predict_svm']
        tmp_file_list = ['{0}-{1}'.format(n, self.fold) for n in tmp_file_list]
        tmp_file_list.append('scale-tmp_svm_train-{0}.out'.format(self.fold))
        for tmp_file in tmp_file_list:
            os.remove(tmp_file)
        logging.info('Independent Test:\n\tAccuracy:{0}\n\tPrecision:{1}\n\tRecall:{2}\n\tF-score:{3}\n'
                     .format(accuracy, precision, recall, fscore))
        return accuracy, precision, recall, fscore


class IntegratedPredictor:
    def __init__(self, label_data, feature_data, fold=0):
        self.label, self.feature = label_data, feature_data
        self.fold = fold

    def select_predictor(self, classifier):
        if classifier == 'svm':
            return self.svm()

    def svm(self):

        def generate_tmp_file(label, feature, file_name='tmp_svm_file'):
            tmp_svm_file = open(file_name, mode='w')
            feature_amount = len(feature[0])
            tmp_output = ''
            for i in range(len(label)):
                tmp_feature = ''
                for n in range(feature_amount):
                    tmp_feature = '{0}\t{1}:{2}'.format(tmp_feature, n + 1, feature[i][n])
                tmp_output = '{0}{1}{2}\n'.format(tmp_output, label[i], tmp_feature)
            tmp_svm_file.write(tmp_output)
            tmp_svm_file.close()
            return feature_amount

        def get_svm_grid_parameter(feature_dimensions, scale_file):
            default_gamma = log(1 / (float(feature_dimensions))) / log(2)
            gamma_start = default_gamma - 1
            gamma_stop = default_gamma + 1
            grid_result = os.popen('svm-grid -log2c -1,1,0.5 -log2g {0},{1},0.5 {2}'
                                   .format(gamma_start, gamma_stop, scale_file)).readlines()[-1].rstrip('\n').split(' ')
            return grid_result[0], grid_result[1]

        feature_dimension_amount = generate_tmp_file(self.label, self.feature, 'tmp_svm_file-{0}'.format(self.fold))
        os.system('svm-scale tmp_svm_file-{0} > tmp_scale_svm_file-{0}'.format(self.fold))
        c_para, g_para = get_svm_grid_parameter(feature_dimension_amount, 'tmp_scale_svm_file-{0}'.format(self.fold))
        train_result = os.popen('svm-train -b 1 -h 0 -v 5 -c {0} -g {1} tmp_scale_svm_file-{2}'
                                .format(c_para, g_para, self.fold)).readlines()[-1]
        train_cv_accuracy = re.search('=\s(.*)%', train_result).group(1)
        tmp_file_list = ['tmp_svm_file-{0}'.format(self.fold), 'tmp_scale_svm_file-{0}'.format(self.fold),
                         'tmp_scale_svm_file-{0}.out'.format(self.fold)]
        for tmp_file in tmp_file_list:
            os.remove(tmp_file)
        return train_cv_accuracy


class IntegrateFeatureSelection:

    def __init__(self, feature_idx, label_data, feature_data, clf, fold):
        self.feature_index = feature_idx
        self.label, self.feature = label_data, feature_data
        self.clf = clf
        self.fold = fold

    def tmp_selected_dataset(self, selected_features):
        tmp_dataset = np.zeros(shape=(len(self.feature), len(selected_features)))
        for feature_count in range(len(selected_features)):
            tmp_dataset[:, feature_count] = self.feature[:, selected_features[feature_count]-1]
        return tmp_dataset

    def backward_selection(self, feature_list, max_acc, removed_features=[]):
        computation_iterations = 0
        logging.info('------Backward Selection Start------\n')
        logging.info('feature list : {0}'.format(feature_list))
        while len(feature_list) > 1:
            local_max_acc = -1
            for each_feature in feature_list:
                tmp_feature_set = [feature for index, feature in enumerate(feature_list) if feature != each_feature]
                tmp_feature_dataset = self.tmp_selected_dataset(tmp_feature_set)
                predict = IntegratedPredictor(self.label, tmp_feature_dataset, self.fold)
                local_acc = predict.select_predictor(self.clf)
                computation_iterations += 1
                logging.debug('\tFeature Index:{0}\tlocal:{1}'.format(each_feature, local_acc))
                if local_acc >= local_max_acc:
                    local_max_acc = local_acc
                    tmp_removed_features = each_feature
            logging.info('\tFeature:{0}\tlocal max:{1}'.format(tmp_removed_features, local_max_acc))
            if local_max_acc >= max_acc:
                max_acc = local_max_acc
                feature_list.remove(tmp_removed_features)
                removed_features.append(tmp_removed_features)
                logging.info('\tGlobal max: {0}\n\t\tRemoved Feature : {1}\n\tFeature Set:{2}'
                             .format(max_acc, tmp_removed_features, feature_list))
            else:
                logging.info('------Backward Selection Complete------\n')
                break
        return feature_list, removed_features, max_acc, computation_iterations

    def sffs_single_list(self, feature_test_uplimit):
        selected_feature_set = []
        computation_iterations = 0
        tested_feature = 0
        max_acc = -1
        start_time = timeit.default_timer()
        while int(float(max_acc)) != 100 and tested_feature < feature_test_uplimit:
            tmp_feature = self.feature_index.pop(0)
            tmp_feature_set = selected_feature_set + [tmp_feature]
            tmp_feature_dataset = self.tmp_selected_dataset(tmp_feature_set)
            predict = IntegratedPredictor(self.label, tmp_feature_dataset, self.fold)
            local_acc = predict.select_predictor(self.clf)
            computation_iterations += 1
            logging.debug('\tFeature Index:{0}\tlocal:{1}'.format(tmp_feature + 1, local_acc))
            if local_acc > max_acc:
                max_acc = local_acc
                selected_feature_set.append(tmp_feature)
            else:
                bc_selected_feature_set, removed_feature, bc_max_acc, bc_computation_iterations \
                    = self.backward_selection(selected_feature_set, max_acc)
                computation_iterations += bc_computation_iterations
                if bc_max_acc > max_acc:
                    max_acc = bc_max_acc
                    selected_feature_set = bc_selected_feature_set
                elif bc_max_acc == max_acc and bc_selected_feature_set != selected_feature_set:
                    selected_feature_set = bc_selected_feature_set
                else:
                    logging.warning('\t--Break!--')
                    break
            tested_feature += 1
            logging.info('Tested Feature: {0}'.format(tested_feature))
        stop_time = timeit.default_timer()
        processing_time = stop_time - start_time
        return max_acc, selected_feature_set, computation_iterations, processing_time

    def modified_sffs_single_list(self, feature_test_uplimit):
        selected_feature_set = []
        computation_iterations = 0
        tested_feature = 0
        max_acc = -1
        start_time = timeit.default_timer()
        while int(float(max_acc)) != 100 and tested_feature < feature_test_uplimit:
            tmp_feature = self.feature_index.pop(0)
            tmp_feature_set = selected_feature_set + [tmp_feature]
            tmp_feature_dataset = self.tmp_selected_dataset(tmp_feature_set)
            predict = IntegratedPredictor(self.label, tmp_feature_dataset, self.fold)
            local_acc = predict.select_predictor(self.clf)
            computation_iterations += 1
            logging.debug('\tFeature Index:{0}\tlocal:{1}'.format(tmp_feature + 1, local_acc))
            if local_acc > max_acc:
                max_acc = local_acc
                selected_feature_set.append(tmp_feature)
            else:
                selected_feature_set, removed_feature, max_acc, bc_computation_iterations \
                    = self.backward_selection(tmp_feature_set, max_acc)
                computation_iterations += bc_computation_iterations
            tested_feature += 1
            logging.info('Tested Feature: {0}'.format(tested_feature))
        stop_time = timeit.default_timer()
        processing_time = stop_time - start_time
        return max_acc, selected_feature_set, computation_iterations, processing_time


if __name__ == '__main__':
    raw_input_file = argv[1]
    selected_classifier = argv[2]
    opts = argv[3:]
    raw_label_data, raw_feature_data = get_data(raw_input_file)
    train_test_fold = train_test_split(raw_label_data, raw_feature_data)
    logging.basicConfig(level=logging.WARNING)
    mode = 0
    i = 0
    while i < len(opts):
        if opts[i] == '-m':
            mode = int(opts[i+1])
        i += 1

    multi_process = multiprocessing.Pool(5)
    for each_fold in train_test_fold:
        each_fold += [selected_classifier, raw_input_file, mode]
    cv_reports = multi_process.map(cv_work, train_test_fold)
    multi_process.close()
    multi_process.join()
    if mode == 0:
        wrapper_name = 'sffs'
    elif mode == 1:
        wrapper_name = 'modified-sffs'

    if not os.path.isdir('control_group'):
        os.makedirs('control_group', 0755)
    for method in cv_reports[0][1]:
        output_file = open('control_group/{0}_{1}-{2}-{3}'
                           .format(method, raw_input_file, selected_classifier, wrapper_name), mode='w')
        for each_report in cv_reports:
            output_file.write(each_report[0][method])
        output_file.close()
    '''
    fold_output = ''
    for each_fold in train_test_fold:
        each_fold += [selected_classifier, raw_input_file]
        tmp_result = cv_work(each_fold)
        fold_output += tmp_result[0]
        output_file_name = tmp_result[1]
    output_file = open(output_file_name, mode='w')
    output_file.write(fold_output)
    output_file.close()
    '''