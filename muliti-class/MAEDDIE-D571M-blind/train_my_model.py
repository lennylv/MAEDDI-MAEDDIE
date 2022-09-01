from datetime import datetime
import time
import argparse
from my_metric import roc_aupr_score, do_compute_metrics

import torch
from sklearn import metrics
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import optim

from data_process import prepare
from sklearn.model_selection import StratifiedKFold
from models import My_Model
from radam import RAdam
from data_process import DDIDataset, DrugDataLoader
import os
import warnings
import random
from torch import nn
from tqdm import tqdm
from data_process import TOTAL_ATOM_FEATS
from pytorchtools import EarlyStopping
warnings.filterwarnings("ignore")
seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_atom_hid = 64
cross_ver_tim = 5
bert_n_heads = 4
bert_n_layers = 4
event_num = 65
n_atom_hid = 64
learn_rating = 0.0003
epo_num = 100
weight_decay_rate=0.01
calssific_loss_weight=6
epoch_changeloss=epo_num//2
n_atom_feats = TOTAL_ATOM_FEATS

class my_loss1(nn.Module):
    def __init__(self):
        super(my_loss1, self).__init__()

        self.criteria1 = torch.nn.CrossEntropyLoss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE1, X_AE2, X_AE4, X_AE5):
        loss = calssific_loss_weight * self.criteria1(X, target) + \
               self.criteria2(inputs.float(), X_AE1) + \
               self.criteria2(inputs.float(), X_AE2) + \
               self.criteria2(inputs.float(), X_AE4) + \
               self.criteria2(inputs.float(), X_AE5)
        return loss

class my_loss2(nn.Module):
    def __init__(self):
        super(my_loss2, self).__init__()

        self.criteria1 = focal_loss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE1, X_AE2, X_AE4, X_AE5):
        loss = calssific_loss_weight * self.criteria1(X, target) + \
               self.criteria2(inputs.float(), X_AE1) + \
               self.criteria2(inputs.float(), X_AE2) + \
               self.criteria2(inputs.float(), X_AE4) + \
               self.criteria2(inputs.float(), X_AE5)
        return loss


class focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(focal_loss, self).__init__()

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1)  # [B * S, 1]
        preds = preds.view(-1, preds.size(-1))  # [B * S, C]

        preds_logsoft = F.log_softmax(preds, dim=1)  # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = loss.mean()

        return loss

class categorical_crossentropy(nn.Module):
    def __init__(self):
        super(categorical_crossentropy, self).__init__()

    def forward(self, pred, label):

        return torch.sum(-label * torch.log(pred))


def My_Model_train(model, x_train, nodeA_feature, nodeB_feature, nodeA_edg, nodeB_edg, y_train, x_test1, nodeA_feature_test1, nodeB_feature_test1, nodeA_edg_test1, nodeB_edg_test1, y_test1,x_test2, nodeA_feature_test2, nodeB_feature_test2, nodeA_edg_test2, nodeB_edg_test2, y_test2, Xfra_train, Xfra_test1, Xfra_test2, event_num):

    model_optimizer = RAdam(model.parameters(), lr=learn_rating, weight_decay=0.0001)
    model = model.to(device)

    x_train = np.vstack((x_train, np.hstack((x_train[:, len(x_train[0]) // 2:], x_train[:, :len(x_train[0]) // 2]))))
    Xfra_train = np.vstack((Xfra_train, Xfra_train))
    y_train = np.hstack((y_train, y_train))
    A = nodeA_feature
    nodeA_feature = nodeA_feature + nodeB_feature
    nodeB_feature = nodeB_feature + A
    A_edg = nodeA_edg
    nodeA_edg = nodeA_edg + nodeB_edg
    nodeB_edg = nodeB_edg + A_edg

    train_dataset = DDIDataset(x_train, nodeA_feature, nodeA_edg, nodeB_feature, nodeB_edg, np.array(y_train),
                               Xfra_train)
    test_dataset = DDIDataset(x_test1, nodeA_feature_test1, nodeA_edg_test1, nodeB_feature_test1, nodeB_edg_test1,
                               np.array(y_test1), Xfra_test1)
    test2_dataset = DDIDataset(x_test2, nodeA_feature_test2, nodeA_edg_test2, nodeB_feature_test2, nodeB_edg_test2,
                               np.array(y_test2), Xfra_test2)
    train_loader = DrugDataLoader(data=train_dataset, batch_size=128, shuffle=True)
    test_loader = DrugDataLoader(data=test_dataset, batch_size=128, shuffle=False)
    test2_loader = DrugDataLoader(data=test2_dataset, batch_size=128, shuffle=False)

    print('Starting training at', datetime.today())
    epoch_all = []
    train_acc_all = []
    test_acc_all = []
    test_auc_roc_all = []
    test_auc_prc_all = []
    test_f1_score_all = []
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(epo_num):
        # if epoch < epoch_changeloss:
        #     my_loss = my_loss1()
        # else:
        #     my_loss = my_loss2()
        my_loss = torch.nn.CrossEntropyLoss()

        start = time.time()
        running_loss = 0.0
        model.train()
        for batch_idx, data in enumerate(train_loader, 0):
            x_batch, y_batch, a, x_fra = data

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_fra = x_fra.to(device)
            a = [tensor.to(device) for tensor in a]

            model_optimizer.zero_grad()
            X = model(x_batch.float(),a,x_fra.float())


            loss=my_loss(X, y_batch)
            loss.backward()
            model_optimizer.step()
            running_loss = running_loss + loss

        model.eval()
        y_pre_all = []
        y_true_all = []
        with torch.no_grad():
            for batch_idx, data in enumerate(train_loader, 0):
                x_batch, y_batch, a, x_fra= data

                x_batch = x_batch.to(device)
                y_true = y_batch.to(device)
                x_fra = x_fra.to(device)
                a = [tensor.to(device) for tensor in a]

                y_pre = model(x_batch.float(), a, x_fra.float())
                y_pre_all.append(F.softmax(y_pre).cpu().numpy())
                y_true_all.append(y_true.cpu().numpy())
            y_pre_all = np.concatenate(y_pre_all)
            y_true_all = np.concatenate(y_true_all)
            y_pre_all = np.argmax(y_pre_all, axis=1)
            train_acc = metrics.accuracy_score(y_true_all, y_pre_all)

        model.eval()
        y_pre_all_test = []
        y_true_all_test = []
        testing_loss = 0.0
        with torch.no_grad():
            for batch_idx, data_test in enumerate(test_loader, 0):
                x_batch_test, y_batch_test, a_test, x_fra = data_test

                x_batch_test = x_batch_test.to(device)
                y_true_test = y_batch_test.to(device)
                x_fra = x_fra.to(device)
                a_test = [tensor.to(device) for tensor in a_test]

                #y_true_test = F.one_hot(y_true_test,num_classes=65)
                y_pre_test= model(x_batch_test.float(), a_test, x_fra.float())
                #loss = my_loss(y_pre_test, y_true_test)
                #testing_loss = testing_loss + loss
                y_pre_all_test.append(F.softmax(y_pre_test).cpu().numpy())
                y_true_all_test.append(y_true_test.cpu().numpy())

            y_pre_all_test_score = np.concatenate(y_pre_all_test)
            y_true_all_test = np.concatenate(y_true_all_test)
            y_pre_all_test = np.argmax(y_pre_all_test_score, axis=1)
        test_acc, test_auc_roc, test_auc_prc, f1_test = do_compute_metrics(y_pre_all_test, y_pre_all_test_score,y_true_all_test)
            # test_acc = metrics.accuracy_score(y_true_all_test, y_pre_all_test)

        model.eval()
        y_pre_all_test = []
        y_true_all_test = []
        testing_loss = 0.0
        with torch.no_grad():
            for batch_idx, data_test in enumerate(test2_loader, 0):
                x_batch_test, y_batch_test, a_test, x_fra = data_test

                x_batch_test = x_batch_test.to(device)
                y_true_test = y_batch_test.to(device)
                x_fra = x_fra.to(device)
                a_test = [tensor.to(device) for tensor in a_test]

                #y_true_test = F.one_hot(y_true_test, num_classes=65)
                y_pre_test = model(x_batch_test.float(), a_test, x_fra.float())
                loss = my_loss(y_pre_test, y_true_test)
                testing_loss = testing_loss + loss
                y_pre_all_test.append(F.softmax(y_pre_test).cpu().numpy())
                y_true_all_test.append(y_true_test.cpu().numpy())

            y_pre_all_test_score = np.concatenate(y_pre_all_test)
            y_true_all_test = np.concatenate(y_true_all_test)
            y_pre_all_test = np.argmax(y_pre_all_test_score, axis=1)
        test2_acc, test2_auc_roc, test2_auc_prc, f1_test2 = do_compute_metrics(y_pre_all_test, y_pre_all_test_score,
                                                                           y_true_all_test)

        epoch_all.append(epoch + 1)
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)
        test_auc_roc_all.append(test_auc_roc)
        test_auc_prc_all.append(test_auc_prc)
        test_f1_score_all.append(f1_test)

        print('Time: %.4f---Epoch [%d]---Train acc:%.6f---Test acc:%.6f---Test_auc_roc:%.6f---Test_auc_prc:%.6f---Test_f1score:%.6f---Test2 acc:%.6f---Test2_auc_roc:%6f---Test2_auc_prc:%6f---Test2_f1score:%.6f' % (time.time() - start,epoch, train_acc, test_acc, test_auc_roc, test_auc_prc, f1_test,test2_acc,test2_auc_roc,test2_auc_prc,f1_test2))
        early_stopping(testing_loss, model)
        if early_stopping.early_stop:
            print('Early Stopping')
            break


    dict = {'epoch': epoch_all, 'train_acc': train_acc_all, 'test_acc': test_acc_all, 'test_auc_roc': test_auc_roc_all,
            'test_auc_prc': test_auc_prc_all, 'test_f1': test_f1_score_all}
    df = pd.DataFrame(dict)
    #df.to_csv(result_path)


def main():
    df_drug = pd.read_csv('./all_data/all_smile.csv')
    train_data = pd.read_csv('./all_data/fold1/train1.csv')
    train_drug_A, train_drug_B, train_type = train_data['d1'], train_data['d2'], train_data['type']
    test1_data = pd.read_csv('./all_data/fold1/test1.csv')
    test1_drug_A, test1_drug_B, test1_type = test1_data['d1'], test1_data['d2'], test1_data['type']
    test2_data = pd.read_csv('./all_data/fold1/test2.csv')
    test2_drug_A, test2_drug_B, test2_type = test2_data['d1'], test2_data['d2'], test2_data['type']

    new_feature_train, new_feature_test1, new_feature_test2, new_label_train, new_label_test1, new_label_test2, train_h, train_t, test1_h, test1_t, test2_h, test2_t, fra_feature_train1, fra_feature_test1, fra_feature_test2, event_num = prepare(
        df_drug, feature_list=["smile", "target", "enzyme"], vector_size=571, train_drug_A=train_drug_A,
        train_drug_B=train_drug_B, train_type=train_type,
        test1_drug_A=test1_drug_A, test1_drug_B=test1_drug_B, test1_type=test1_type,
        test2_drug_A=test2_drug_A, test2_drug_B=test2_drug_B, test2_type=test2_type)
    print("dataset len", len(new_feature_train))

    new_feature_train = new_feature_train.astype(np.float32)
    new_feature_test1 = new_feature_test1.astype(np.float32)
    new_feature_test2 = new_feature_test2.astype(np.float32)
    new_label_train = new_label_train.astype(np.float32)
    new_label_test1 = new_label_test1.astype(np.float32)
    new_label_test2 = new_label_test2.astype(np.float32)
    fra_feature_train1 = fra_feature_train1.astype(np.float32)
    fra_feature_test1 = fra_feature_test1.astype(np.float32)
    fra_feature_test2 = fra_feature_test2.astype(np.float32)

    node_A = []
    node_A_list = []
    node_B = []
    node_B_list = []
    node_A_test1 = []
    node_A_list_test1 = []
    node_B_test1 = []
    node_B_list_test1 = []
    node_A_test2 = []
    node_A_list_test2 = []
    node_B_test2 = []
    node_B_list_test2 = []

    for i in tqdm(range(len(train_h))):
        node_A.append(list(train_h[i][0]))
        node_B.append(list(train_t[i][0]))
        node_A_list.append(list(train_h[i][1]))
        node_B_list.append(list(train_t[i][1]))
    for j in tqdm(range(len(test1_h))):
        node_A_test1.append(list(test1_h[j][0]))
        node_B_test1.append(list(test1_t[j][0]))
        node_A_list_test1.append(list(test1_h[j][1]))
        node_B_list_test1.append(list(test1_t[j][1]))
    for l in tqdm(range(len(test2_h))):
        node_A_test2.append(list(test2_h[l][0]))
        node_B_test2.append(list(test2_t[l][0]))
        node_A_list_test2.append(list(test2_h[l][1]))
        node_B_list_test2.append(list(test2_t[l][1]))

    model = My_Model(len(new_feature_train[0]), bert_n_heads, bert_n_layers, event_num, n_atom_feats, n_atom_hid, heads_out_feat_params=[32, 32, 32, 32], blocks_params=[4, 4, 4, 4], fra_dim=len(fra_feature_train1[0]))



    My_Model_train(model, new_feature_train, node_A, node_B, node_A_list, node_B_list, new_label_train,
                   new_feature_test1, node_A_test1, node_B_test1,
                   node_A_list_test1, node_B_list_test1, new_label_test1, new_feature_test2, node_A_test2, node_B_test2,
                   node_A_list_test2, node_B_list_test2, new_label_test2, fra_feature_train1, fra_feature_test1,
                   fra_feature_test2, 65)





main()



