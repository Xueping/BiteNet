from utils.icd9_ontology import ICD_Ontology as icd
from utils.icd9_ontology import CCS_Ontology as ccs
from abc import ABCMeta
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, auc
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from utils.configs import cfg


class EvaluationTemplate(metaclass=ABCMeta):

    def __init__(self, model, logging):
        self.logging = logging
        if model is not None:
            self.model = model
            self.reverse_dict = model.reverse_dict
            self.dictionary = dict(zip(model.reverse_dict.values(), model.reverse_dict.keys()))
            self.verbose = model.verbose
            self.valid_samples = model.valid_samples
            self.top_k = model.top_k

    # @abstractmethod
    def get_clustering_nmi(self, sess, ground_truth):
        pass

    # @abstractmethod
    def get_nns_p_at_top_k(self, sess, ground_truth):
        pass

    # @abstractmethod
    def get_nns_pairs_count(self, ground_truth):
        pass

    @staticmethod
    def metric_pred(y_true, probs, y_pred):
        [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
        # print(TN, FP, FN, TP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (FP + TN)
        precision = TP / (TP + FP)
        sensitivity = recall = TP / (TP + FN)
        # f_score = 2 * TP / (2 * TP + FP + FN)

        # calculate AUC
        roc_auc = roc_auc_score(y_true, probs)
        # print('roc_auc: %.4f' % roc_auc)
        # calculate roc curve
        # fpr, tpr, thresholds = roc_curve(y_true, probs)

        # calculate precision-recall curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probs)
        # calculate F1 score
        f_score = f1_score(y_true, y_pred)
        # calculate precision-recall AUC

        pr_auc = auc(recall_curve, precision_curve)
        accuracy = round(accuracy, 4)
        precision = round(precision, 4)
        sensitivity = round(sensitivity, 4)
        specificity = round(specificity, 4)
        f_score = round(f_score, 4)
        pr_auc = round(pr_auc, 4)
        roc_auc = round(roc_auc, 4)

        return [accuracy, precision, sensitivity, specificity, f_score, pr_auc, roc_auc]

    @staticmethod
    def recall_top(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
        recall = list()
        for i in range(len(y_pred)):
            this_one = list()
            codes = y_true[i]
            tops = y_pred[i]
            for rk in rank:
                length = len(set(codes))
                if length > rk:
                    length = rk
                this_one.append(len(set(codes).intersection(set(tops[:rk])))*1.0/length)
            recall.append(this_one)
        return np.round((np.array(recall)).mean(axis=0), decimals=4).tolist(), \
               np.round((np.array(recall)).std(axis=0), decimals=4).tolist()

    @staticmethod
    def code_level_top(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
        recall = list()
        for i in range(len(y_pred)):
            this_one = list()
            codes = y_true[i]
            tops = y_pred[i]
            for rk in rank:
                this_one.append(len(set(codes).intersection(set(tops[:rk]))) * 1.0 / rk)
            recall.append(this_one)
        return np.round((np.array(recall)).mean(axis=0), decimals=4).tolist()


class ConceptEvaluation(object):
    def __init__(self, dataset, logging):
        self.icd_file = cfg.icd_file
        self.ccs_file = cfg.ccs_file
        self.top_k = cfg.top_k
        self.logging = logging
        self.dictionary = dataset.dictionary
        self.reverse_dict = dataset.reverse_dictionary

    def get_clustering_nmi(self, embeddings, ground_truth):
        # embeddings = sess.run(self.model.final_weights)

        if ground_truth == 'ICD':
            icd9 = icd(self.icd_file, True)
        else:
            icd9 = ccs(self.ccs_file)

        # get codes and index for diagnosis codes
        dx_codes = []
        index_codes = []  # index of valid diagnosis codes
        for k, v in self.reverse_dict.items():
            if v.startswith('D_'):
                dx_codes.append(v[2:])
                index_codes.append(int(k))

        # Label high level category (0:18) for each code
        dx_labels = np.empty(len(dx_codes), int)
        dx_index = 0
        for code in dx_codes:
            dx_labels[dx_index] = icd9.getRootLevel(code)
            dx_index += 1

        dx_weights = embeddings[index_codes]

        dx_uni_labels = np.unique(dx_labels).shape[0]
        k_means = KMeans(n_clusters=dx_uni_labels, random_state=42).fit(dx_weights)

        nmi = metrics.normalized_mutual_info_score(dx_labels, k_means.labels_)
        nmi_round = round(nmi, 4)

        if ground_truth == 'ICD':
            log_str = "number of dx_labels in ICD9: %s" % dx_uni_labels
            self.logging.add(log_str)
            log_str = "ICD, NMI Score:%s" % nmi_round
            self.logging.add(log_str)
        else:
            log_str = "number of dx_labels in CCS: %s" % dx_uni_labels
            self.logging.add(log_str)
            log_str = "CCS, NMI Score:%s" % nmi_round
            self.logging.add(log_str)

        return nmi_round

    def get_nns_p_at_top_k(self, embeddings, ground_truth, top_k):
        # similarity = sess.run(self.model.final_wgt_sim)
        if ground_truth == 'ICD':
            icd9 = icd(self.icd_file, True)
        else:
            icd9 = ccs(self.ccs_file)

        # get codes and index for diagnosis codes
        dx_codes = []
        index_codes = []  # index of valid diagnosis codes
        for k, v in self.reverse_dict.items():
            if v.startswith('D_'):
                dx_codes.append(v[2:])
                index_codes.append(int(k))

        dx_weights = embeddings[index_codes]

        valid_index = list(range(cfg.valid_size))
        # valid_codes = dx_codes[valid_index]
        valid_weights = dx_weights[valid_index]
        similarity = np.matmul(valid_weights, np.transpose(dx_weights))

        total_precision = 0.
        for i in range(cfg.valid_size):
            valid_code = dx_codes[i]
            valid_label = icd9.getRootLevel(valid_code)
            nearest = (-similarity[i, :]).argsort()[1:top_k+1]
            no_same_cat = 0.
            actual_k = 0
            for k in range(top_k):
                close_word = dx_codes[nearest[k]]
                actual_k += 1
                close_label = icd9.getRootLevel(close_word)
                if valid_label == close_label:
                    no_same_cat += 1
            if actual_k > 0:
                total_precision += no_same_cat/actual_k

        evg_precision = round(total_precision/cfg.valid_size, 4)

        if ground_truth == 'ICD':
            log_str = "ICD NNS P@%s Score:%s" % (top_k, evg_precision)
        else:
            log_str = "CCS NNS P@%s Score:%s" % (top_k, evg_precision)
        self.logging.add(log_str)

        return evg_precision

    def get_nns_pairs_count(self, ground_truth):

        code_list = list(self.reverse_dict.values())[1:]
        dx_codes = list()
        tx_codes = list()
        for code in code_list:
            if code.startswith('D_'):
                dx_codes.append(code)
            else:
                tx_codes.append(code)

        if ground_truth == 'ICD':
            icd9 = icd(self.icd_file, True)
        else:
            icd9 = ccs(self.ccs_file)

        label_cnt_dict = {}
        for dx in dx_codes:
            label = icd9.getRootLevel(dx[2:])
            if label in label_cnt_dict:
                label_cnt_dict[label] += 1
            else:
                label_cnt_dict[label] = 1

        no_cat = len(label_cnt_dict)
        total_pairs = 0
        for k, v in label_cnt_dict.items():
            total_pairs += v * (v-1)/2

        return no_cat, total_pairs
