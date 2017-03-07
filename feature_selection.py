__author__ = 'HSUAN-WEI YANG'

from scipy.stats.stats import pearsonr, kendalltau, spearmanr
from sklearn.feature_selection import chi2
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.decomposition import PCA as sk_pca
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score, SPEC, fisher_score
from sklearn.ensemble import RandomForestRegressor
from minepy import MINE
from sys import argv
import os
import re
import logging
import numpy as np
import progressbar as pb


class SupervisedFs:

    def __init__(self,  input_data):
        # initialize the options' parameter
        self.options = {'p': 0, 'r': 0, 'm': 0, 'c': 0, 'k': 0, 's': 0, 'f': 0}
        logging.basicConfig(level=logging.INFO)
        self.data_label, self.data_feature = self.modify_input_data(input_data)
        self.data_name = input_data.split('/')[-1]

    def parse_options(self, options):
        for each_option in options:
            if each_option[1:] not in ('p', 'r', 'm', 'c', 'k', 's', 'a', 'f'):
                raise ValueError('Options Error')
            if each_option[1:] == 'a':
                for op_value in self.options.keys():
                    self.options[op_value] = 1
                break
            self.options[each_option[1:]] = 1

    @staticmethod
    def modify_input_data(dataset_pathname):
        logging.info('     Modifying Input Data    ')
        if not os.path.exists(dataset_pathname):
            raise IOError('Dataset not found')
        # Two kinds of input data format are acceptable and defined as follows:
        # First  : label,feature1,feature2,....
        # Second : label 1:feature1 2
        try:
            # For first kind
            file_input = np.loadtxt(dataset_pathname, delimiter=',')
        except ValueError:
            # For second kind
            raw_data = open(dataset_pathname, mode='r').readlines()
            file_input = []
            widget = ['Processing                       : ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()),
                      ' ', pb.ETA()]
            timer = pb.ProgressBar(widgets=widget, maxval=len(raw_data)).start()
            n = 0
            for each_line in raw_data:
                each_line = each_line.rstrip()
                line_data = re.split(r'\t|\s', each_line)
                tmp_modified_line = [float(line_data[0])]
                for i in range(1, len(line_data), 1):
                    tmp_modified_line.append(float(line_data[i].split(':')[-1]))
                file_input.append(tmp_modified_line)
                timer.update(n)
                n += 1
            timer.finish()
            file_input = np.asarray(file_input)
        except:
            raise IOError('Data format not suitable')
        finally:
            data_label = file_input[:, 0]
            data_feature = file_input[:, 1:]
            logging.info('     Modifying Input Data ==> Done    \n')
            return data_label, data_feature

    @staticmethod
    def normalize_feature(raw_data):
        logging.info('     Normalizing Input Data    ')
        total_amount = raw_data[0].size
        for i in range(total_amount):
            max_value = raw_data[:, i].max()
            min_value = raw_data[:, i].min()
            if (max_value-min_value) == 0:
                raw_data[:, i] = 0
            else:
                raw_data[:, i] = (raw_data[:, i]-min_value)/(max_value-min_value)
        logging.info('     Normalizing Input Data ==> Done   \n')
        return raw_data

    @staticmethod
    def output_file(input_filename, selection_result, output_dir='./fs_result/supervised'):
        logging.info('     Writing Output File     ')
        #if not os.path.isdir('./fs_result'):
        #    os.makedirs('./fs_result', 0755)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, 0755)
        for key in selection_result.keys():
            outfile = open('{0}/{1}_{2}'.format(output_dir, key, input_filename), mode='w')
            out_str = ''
            for result in selection_result[key]:
                out_str += '{0}\t{1}\n'.format(result[1], result[0])
            outfile.write(out_str)
            outfile.close()
        logging.info('     Writing Output File  ==>  Output directory:{0} ==> Done   '.format(output_dir))

    def utilize_selection_method(self, options):
        self.parse_options(options)
        normalize_feature = self.normalize_feature(self.data_feature)
        feature_amount = len(self.data_feature[0])
        selection_result = {}
        logging.info('     Supervised Feature Selection : Start')
        if self.options['p'] == 1:
            widget = ['Calculating Pearson Correlation  : ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()),
                      ' ', pb.ETA()]
            timer = pb.ProgressBar(widgets=widget, maxval=feature_amount).start()
            pearson_corr = []
            for n in range(0, feature_amount):
                tmp_pearson = pearsonr(normalize_feature[:, n], self.data_label)
                pearson_corr.append([abs(tmp_pearson[0]), n+1])
                timer.update(n)
            timer.finish()
            selection_result['pearson-correlation'] = sorted(pearson_corr, reverse=True)

        if self.options['r'] == 1:
            widget = ['Calculating Random Forest        : ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()),
                      ' ', pb.ETA()]
            timer = pb.ProgressBar(widgets=widget, maxval=feature_amount).start()
            rf = RandomForestRegressor(n_estimators=20, max_depth=4)
            #rf.fit(normalize_feature, self.data_label)
            #rf_feature_score = rf.feature_importances_
            random_forest = []
            for n in range(0, feature_amount):
                score = cross_val_score(rf, normalize_feature[:, n:n+1], self.data_label, scoring="r2",
                                        cv=ShuffleSplit(len(normalize_feature), 3, .3))
                random_forest.append([round(np.mean(score), 3), n+1])
                #random_forest.append([rf_feature_score[n], n+1])
                timer.update(n)
            timer.finish()
            selection_result['random-forest'] = sorted(random_forest, reverse=True)

        if self.options['m'] == 1:
            widget = ['Calculating Mutual Information   : ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()),
                      ' ', pb.ETA()]
            timer = pb.ProgressBar(widgets=widget, maxval=feature_amount).start()
            mutual_information = []
            mine = MINE()
            for n in range(0, feature_amount):
                mine.compute_score(normalize_feature[:, n], self.data_label)
                mutual_information.append([mine.mic(), n+1])
                timer.update(n)
            timer.finish()
            selection_result['mutual-information'] = sorted(mutual_information, reverse=True)

        if self.options['c'] == 1:
            widget = ['Calculating Chi Squire           : ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()),
                      ' ', pb.ETA()]
            timer = pb.ProgressBar(widgets=widget, maxval=feature_amount).start()
            chi_squire = []
            compute_chi2 = chi2(normalize_feature, self.data_label)[0]
            for n in range(0, feature_amount):
                chi_squire.append([compute_chi2[n], n+1])
                timer.update(n)
            timer.finish()
            selection_result['chi-squire'] = sorted(chi_squire, reverse=False)

        if self.options['k'] == 1:
            widget = ['Calculating Kendall Correlation  : ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()),
                      ' ', pb.ETA()]
            timer = pb.ProgressBar(widgets=widget, maxval=feature_amount).start()
            kendall_correlation = []
            for n in range(0, feature_amount):
                tmp_kendall = kendalltau(normalize_feature[:, n], self.data_label)
                kendall_correlation.append([tmp_kendall[0], n+1])
                timer.update(n)
            timer.finish()
            selection_result['kendall-correlation'] = sorted(kendall_correlation, reverse=True)

        if self.options['s'] == 1:
            widget = ['Calculating Spearman Correlation : ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()),
                      ' ', pb.ETA()]
            timer = pb.ProgressBar(widgets=widget, maxval=feature_amount).start()
            spearman_corr = []
            for n in range(0, feature_amount):
                tmp_spearman = spearmanr(normalize_feature[:, n], self.data_label)
                spearman_corr.append([abs(tmp_spearman[0]), n+1])
                timer.update(n)
            timer.finish()
            selection_result['spearman-correlation'] = sorted(spearman_corr, reverse=True)

        if self.options['f'] == 1:
            logging.info('   -----Calculating Fisher score---- ')
            f_score = fisher_score.fisher_score(normalize_feature, self.data_label)
            fisher = []
            for n in range(0, feature_amount):
                fisher.append([f_score[n], n+1])
            selection_result['fisher-score'] = sorted(fisher, reverse=True)
            logging.info('   -----Calculating Fisher score---- ==> Done')
        return selection_result


class UnsupervisedFs:
    def __init__(self,  dataset):
        # initialize the options' parameter
        self.options = {'p': 0, 'l': 0, 'v': 0, 's': 0}
        logging.basicConfig(level=logging.INFO)
        self.data_label, self.data_feature = SupervisedFs.modify_input_data(dataset)
        self.data_name = dataset.split('/')[-1]

    def parse_options(self, options):
        for each_option in options:
            if each_option[1:] not in ('p', 'v', 'l', 's', 'a'):
                raise ValueError('Options Error')
            if each_option[1:] == 'a':
                for op_value in self.options.keys():
                    self.options[op_value] = 1
                break
            self.options[each_option[1:]] = 1

    def modify_input_file(self, dataset_pathname):
        logging.info('     Modifying Input Data    ')
        if not os.path.exists(dataset_pathname):
            raise IOError('Dataset not found')
        # Two kinds of input data format are acceptable and defined as follows:
        # First  : feature1,feature2,....
        # Second : 1:feature1 2:feature2 ......
        try:
            # For first kind
            data_feature = np.loadtxt(dataset_pathname, delimiter=',')
        except ValueError:
            # For second kind
            raw_data = open(dataset_pathname, mode='r').readlines()
            file_input = []
            widget = ['Processing                       : ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()),
                      ' ', pb.ETA()]
            timer = pb.ProgressBar(widgets=widget, maxval=len(raw_data)).start()
            n = 0
            for each_line in raw_data:
                each_line = each_line.rstrip()
                line_data = re.split(r'\t|\s', each_line)
                tmp_modified_line = []
                for i in range(1, len(line_data), 1):
                    tmp_modified_line.append(float(line_data[i].split(':')[-1]))
                file_input.append(tmp_modified_line)
                timer.update(n)
                n += 1
            timer.finish()
            data_feature = np.asarray(file_input)
        except:
            raise IOError('Data format not suitable')
        finally:
            logging.info('     Modifying Input Data ==> Done    \n')
            return data_feature

    def utilize_selection_method(self, options):
        logging.info('     Unsupervised Feature Selection : Start')
        self.parse_options(options)
        normalize_feature = SupervisedFs.normalize_feature(self.data_feature)
        feature_amount = len(self.data_feature[0])
        selection_result = {}

        if self.options['v'] == 1:
            widget = ['Calculating Variance             : ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()),
                      ' ', pb.ETA()]
            timer = pb.ProgressBar(widgets=widget, maxval=feature_amount).start()
            variance = []
            for n in range(0, feature_amount):
                variance.append([np.var(normalize_feature[:, n]), n+1])
                timer.update(n)
            timer.finish()
            selection_result['variance'] = sorted(variance, reverse=True)

        if self.options['l'] == 1:
            logging.info('   -----Calculating Laplacian score---- ')
            kwargs_w = {'metric': 'euclidean', 'neighbor': 'knn', 'weight_mode': 'heat_kernel', 'k': 5, 't': 1}
            W = construct_W.construct_W(self.data_feature, **kwargs_w)
            score = lap_score.lap_score(self.data_feature, W=W)
            lap = []
            for n in range(0, feature_amount):
                lap.append([score[n], n+1])
            selection_result['laplacian'] = sorted(lap, reverse=False)
            logging.info('   -----Calculating Laplacian score---- ==> Done')

        if self.options['s'] == 1:
            logging.info('   -----Calculating Spectral score---- ')
            kwargs_w = {'metric': 'euclidean', 'neighbor': 'knn', 'weight_mode': 'heat_kernel', 'k': 5, 't': 1}
            W = construct_W.construct_W(self.data_feature, **kwargs_w)
            kwargs_s = {'style': 2, 'W': W}
            score = SPEC.spec(self.data_feature, **kwargs_s)
            spec = []
            for n in range(0, feature_amount):
                spec.append([score[n], n+1])
            selection_result['spectral'] = sorted(spec, reverse=True)
            logging.info('   -----Calculating Spectral score---- ==> Done')
        return selection_result

    def generate_pca(self, require_dim, output_dir='./fs_result/'):
        logging.info('   -----Calculating PCA---- ')
        out_file = open('{0}pca_{1}_{2}'.format(output_dir, self.data_name, require_dim), mode='w')
        pca = sk_pca(n_components=require_dim)
        new_feature = pca.fit_transform(self.data_feature)
        for each_sample in range(len(new_feature)):
            combine_str = ''
            for each_feature in range(require_dim):
                combine_str += str(each_feature+1) + ':' + str(new_feature[each_sample][each_feature]) + '\t'
            out_file.write(combine_str.rstrip('\t') + '\n')
        out_file.close()
        logging.info('   -----Calculating PCA----- ==>  Output directory:{0}  ==> Done'.format(output_dir))

    def output_file(self, input_file_name, selection_result, output_dir='./fs_result/unsupervised'):
        SupervisedFs.output_file(input_file_name, selection_result, output_dir)


if __name__ == '__main__':

    input_file = argv[1]
    input_options = argv[2:]
    logging.basicConfig(level=logging.WARNING)
    i = 0
    output_directory = None
    unsup_index = None
    sup_index = None
    while i < len(input_options):
        if input_options[i] == '-o':
            output_directory = argv[i+3]
        elif input_options[i] == '-un':
            unsup_index = i+2
        elif input_options[i] == '-sup':
            sup_index = i+2
        i += 1
    if unsup_index is not None and sup_index is not None:
        unsup_selection = UnsupervisedFs(input_file)
        sup_selection = SupervisedFs(input_file)
        if unsup_index < sup_index:
            unsup_selection_result = unsup_selection.utilize_selection_method(argv[unsup_index+1:sup_index])
            sup_selection_result = sup_selection.utilize_selection_method(argv[sup_index+1:])
        else:
            unsup_selection_result = unsup_selection.utilize_selection_method(argv[unsup_index+1:])
            sup_selection_result = sup_selection.utilize_selection_method(argv[sup_index+1:unsup_index])
    elif unsup_index is not None and sup_index is None:
        unsup_selection = UnsupervisedFs(input_file)
        unsup_selection_result = unsup_selection.utilize_selection_method(argv[unsup_index+1:])
    elif unsup_index is None and sup_index is not None:
        sup_selection = SupervisedFs(input_file)
        sup_selection_result = sup_selection.utilize_selection_method(argv[sup_index+1:])
    else:
        print 'No method required'

    if output_directory is None:
        if unsup_index is not None:
            unsup_selection.output_file(input_file, unsup_selection_result)
        if sup_index is not None:
            sup_selection.output_file(input_file, sup_selection_result)

    else:
        if unsup_index is not None:
            unsup_selection.output_file(input_file, unsup_selection_result, output_directory)
        if sup_index is not None:
            sup_selection.output_file(input_file, sup_selection_result, output_directory)



