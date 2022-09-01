
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.preprocessing import label_binarize
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

def do_compute_metrics(probas_pred, y_score,  target):

    y_one_hot = label_binarize(target, classes=np.arange(65))
    acc = metrics.accuracy_score(target, probas_pred)
    auc_roc = metrics.roc_auc_score(y_one_hot, y_score, average='micro')
    f1_score = metrics.f1_score(target, probas_pred, average='micro')
    auc_prc = roc_aupr_score(y_one_hot, y_score, average='micro')

    return acc, auc_roc, auc_prc, f1_score