
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def do_compute_metrics(pred_type, pred_score,  y_test, event_num):
    all_eval_type = 6
    each_eval_type = 6

    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)

    acc = accuracy_score(y_test, pred_type)
    prc = roc_aupr_score(y_one_hot, pred_score, average='micro')
    auc = roc_auc_score(y_one_hot, pred_score, average='micro')
    f1 = f1_score(y_test, pred_type, average='macro')
    precision = precision_score(y_test, pred_type, average='macro')
    recal = recall_score(y_test, pred_type, average='macro')

    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average=None)
        #result_eve[i, 2] = 0.0
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')

    return acc, prc, auc, f1, precision, recal, result_eve

def do_compute_metrics_1(pred_type, pred_score,  y_test, event_num):

    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))

    acc = accuracy_score(y_test, pred_type)
    prc = roc_aupr_score(y_one_hot, pred_score, average='micro')
    auc = roc_auc_score(y_one_hot, pred_score, average='micro')
    f1 = f1_score(y_test, pred_type, average='macro')
    precision = precision_score(y_test, pred_type, average='macro')
    recal = recall_score(y_test, pred_type, average='macro')

    return acc, prc, auc

def class_divide(y, y_predict):
    length = len(y)
    true_dic = {}
    pre_true_dic = {}
    for i in range(65):
        true_dic[i] = 0
        pre_true_dic[i] = 0
    for i in range(length):
        true_dic[y[i]] = true_dic[y[i]] + 1
        if y[i] == y_predict[i]:
            pre_true_dic[y[i]] = pre_true_dic[y[i]] + 1
    true_dic_list = list(true_dic.values())
    pre_true_dic_list = list(pre_true_dic.values())
    res_divide = []
    for i in range(65):
        if true_dic_list[i] == 0:
            res_divide.append(0)
        else:
            res_divide.append(pre_true_dic_list[i] / true_dic_list[i])
    res_65 = sum(pre_true_dic_list)/sum(true_dic_list)


    return res_divide,res_65