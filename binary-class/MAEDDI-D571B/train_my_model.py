from datetime import datetime
import time
import torch
from sklearn import metrics
import pandas as pd
import numpy as np
import torch.nn.functional as F
from data_process import prepare
from sklearn.model_selection import StratifiedKFold
from models import My_Model
from radam import RAdam
from data_process import DDIDataset, DrugDataLoader
import os
import warnings
import random
from data_process import TOTAL_ATOM_FEATS


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
n_atom_hid = 64
learn_rating = 0.00001
epo_num = 150
weight_decay_rate=0.0001
calssific_loss_weight=6
epoch_changeloss=epo_num//4
n_atom_feats = TOTAL_ATOM_FEATS
n_atom_feats = TOTAL_ATOM_FEATS

event_num = 2
batch_size = 256


def My_Model_train(model, x_train, nodeA_feature, nodeB_feature, nodeA_edg, nodeB_edg, y_train, x_test, nodeA_feature_test, nodeB_feature_test, nodeA_edg_test, nodeB_edg_test, y_test, event_num, result_path, Xfra_train, Xfra_test):

    model_optimizer = RAdam(model.parameters(), lr=learn_rating, weight_decay=weight_decay_rate)
    model = model.to(device)

    x_train = np.vstack((x_train, np.hstack((x_train[:, len(x_train[0]) // 2:], x_train[:, :len(x_train[0]) // 2]))))
    Xfra_train = np.vstack((Xfra_train,Xfra_train))
    y_train = np.hstack((y_train, y_train))
    A = nodeA_feature
    nodeA_feature = nodeA_feature + nodeB_feature
    nodeB_feature = nodeB_feature + A
    A_edg = nodeA_edg
    nodeA_edg = nodeA_edg + nodeB_edg
    nodeB_edg = nodeB_edg + A_edg


    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(Xfra_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)
    np.random.seed(seed)
    np.random.shuffle(nodeA_feature)
    np.random.seed(seed)
    np.random.shuffle(nodeA_edg)
    np.random.seed(seed)
    np.random.shuffle(nodeB_feature)
    np.random.seed(seed)
    np.random.shuffle(nodeB_edg)


    train_dataset = DDIDataset(x_train, nodeA_feature, nodeA_edg, nodeB_feature, nodeB_edg, np.array(y_train),Xfra_train)
    test_dataset = DDIDataset(x_test,nodeA_feature_test, nodeA_edg_test, nodeB_feature_test, nodeB_edg_test,np.array(y_test),Xfra_test)
    train_loader = DrugDataLoader(data=train_dataset, batch_size=128, shuffle=True)
    test_loader = DrugDataLoader(data=test_dataset, batch_size=128, shuffle=False)

    print('Starting training at', datetime.today())
    epoch_all = []
    train_acc_all = []
    test_acc_all = []
    test_auc_roc_all = []
    test_f1_score_all = []


    for epoch in range(epo_num):

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
                y_pre = model(x_batch.float(),a, x_fra.float())
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
                y_pre_test = model(x_batch_test.float(), a_test, x_fra.float())
                loss = my_loss(y_pre_test, y_true_test)
                testing_loss = testing_loss + loss
                y_pre_all_test.append(F.softmax(y_pre_test).cpu().numpy())
                y_true_all_test.append(y_true_test.cpu().numpy())

            y_pre_all_test_score = np.concatenate(y_pre_all_test)
            y_true_all_test = np.concatenate(y_true_all_test)
            y_pre_all_test = np.argmax(y_pre_all_test_score, axis=1)

            test_acc = metrics.accuracy_score(y_true_all_test, y_pre_all_test)
            test_f1 = metrics.f1_score(y_true_all_test, y_pre_all_test)
            y_true_all_test = y_true_all_test.astype(np.int64)
            y_true_all_test = torch.from_numpy(y_true_all_test)
            y_true_all_test = F.one_hot(y_true_all_test, num_classes=2)
            y_true_all_test = np.array(y_true_all_test)
            test_roc = metrics.roc_auc_score(y_true_all_test, y_pre_all_test_score)

        epoch_all.append(epoch + 1)
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)
        test_auc_roc_all.append(test_roc)
        test_f1_score_all.append(test_f1)


        print('Time: %.4f---Epoch [%d]---Train acc:%.6f---Test acc:%.6f---Test_auc_roc:%.6f---Test_f1score:%.6f ' % (time.time() - start,epoch, train_acc, test_acc, test_roc,test_f1))

    PATH = 'state_dict_model.pth'
    torch.save(model.state_dict(),PATH)
    dict = {'epoch': epoch_all, 'train_acc': train_acc_all, 'test_acc': test_acc_all, 'test_auc_roc': test_auc_roc_all,'test_f1': test_f1_score_all}
    df = pd.DataFrame(dict)
    df.to_csv(result_path)


def main():
    df_drug = pd.read_csv('./data/all_smile.csv')
    extraction = pd.read_csv('./data/new_test.csv')
    drugA = extraction['d1']
    drugB = extraction['d2']
    type = extraction['type']
    smileA = extraction['smile1']
    smileB = extraction['smile2']

    new_feature, new_label, pos_h, pos_t, drugA, drugB, event_num, fra_feature = prepare(df_drug,feature_list=["smile", "target", "enzyme"],type=type, drugA=drugA, drugB=drugB, smileA=smileA,smileB=smileB)
    print("dataset len", len(new_feature))
    np.random.seed(seed)
    np.random.shuffle(new_feature)
    np.random.seed(seed)
    np.random.shuffle(new_label)
    np.random.seed(seed)
    np.random.shuffle(pos_h)
    np.random.seed(seed)
    np.random.shuffle(pos_t)
    np.random.seed(seed)
    np.random.shuffle(drugA)
    np.random.seed(seed)
    np.random.shuffle(drugB)
    np.random.seed(seed)
    np.random.shuffle(fra_feature)

    new_feature = new_feature.astype(np.longlong)
    new_label = new_label.astype(np.longlong)
    feature = new_feature
    label = new_label
    fra = fra_feature.astype(np.longlong)
    skf = StratifiedKFold(n_splits=cross_ver_tim)

    path_num = 0

    for train_index, test_index in skf.split(feature, label):
        path_num = path_num + 1
        model = My_Model(len(feature[0]), bert_n_heads, bert_n_layers, event_num, n_atom_feats, n_atom_hid, heads_out_feat_params=[32, 32, 32, 32], blocks_params=[4, 4, 4, 4], fra_dim=len(fra[0]))
        node_A = []
        node_A_list = []
        node_B = []
        node_B_list = []
        node_A_test = []
        node_A_list_test = []
        node_B_test = []
        node_B_list_test = []
        for i in train_index:
            node_A.append(list(pos_h[i][0]))
            node_B.append(list(pos_t[i][0]))
            node_A_list.append(list(pos_h[i][1]))
            node_B_list.append(list(pos_t[i][1]))
        for j in test_index:
            node_A_test.append(list(pos_h[j][0]))
            node_B_test.append(list(pos_t[j][0]))
            node_A_list_test.append(list(pos_h[j][1]))
            node_B_list_test.append(list(pos_t[j][1]))
        X_train, X_test = feature[train_index], feature[test_index]
        Xfra_train, Xfra_test = fra[train_index], fra[test_index]
        y_train, y_test = label[train_index], label[test_index]

        result_path = './two-result' + str(path_num) + '.csv'

        My_Model_train(model, X_train, node_A, node_B, node_A_list, node_B_list, y_train, X_test, node_A_test, node_B_test, node_A_list_test, node_B_list_test, y_test, 2, result_path,Xfra_train, Xfra_test)






main()



