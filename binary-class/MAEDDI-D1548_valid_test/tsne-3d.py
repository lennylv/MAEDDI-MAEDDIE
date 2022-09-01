import itertools
from collections import defaultdict
from operator import neg
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data,Batch
from rdkit import Chem
import pandas as pd
import numpy as np
from pandas import DataFrame
from smiles2vector import smiles2vector
from tqdm import tqdm
from models import My_Model
from sklearn import metrics
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


df_drugs_smiles = pd.read_csv('data/AMDE_drug.csv')

DRUG_TO_INDX_DICT = {drug_id: indx for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}

drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]

# Gettings information and features of atoms
ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])
AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
max_valence = max(max_valence, 9)
AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(),
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return results

def get_atom_features(atom, mode='one_hot'):

    if mode == 'one_hot':
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            one_of_k_encoding_unk(atom.GetDegree(), AVAILABLE_ATOM_DEGREES),
            one_of_k_encoding_unk(atom.GetTotalNumHs(), AVAILABLE_ATOM_TOTAL_HS),
            one_of_k_encoding_unk(atom.GetImplicitValence(), AVAILABLE_ATOM_VALENCE),
            torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
        ])
    else:
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            torch.tensor([atom.GetDegree()]).float(),
            torch.tensor([atom.GetTotalNumHs()]).float(),
            torch.tensor([atom.GetImplicitValence()]).float(),
            torch.tensor([atom.GetIsAromatic()]).float()
        ])

    return atom_feature


def get_mol_edge_list_and_feat_mtx(mol_graph):
    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    features.sort()  # to make sure that the feature matrix is aligned according to the idx of the atom
    _, features = zip(*features)
    features = np.stack(features)

    edge_list = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = np.concatenate([edge_list, edge_list[:, [1, 0]]], axis=0) if len(edge_list) else edge_list

    return undirected_edge_list.T, features

def create_graph_data(id):
    edge_index = MOL_EDGE_LIST_FEAT_MTX[id][0]
    features = MOL_EDGE_LIST_FEAT_MTX[id][1]

    return [features, edge_index]#Data(x=features, edge_index=edge_index)

def prepare(df_drug, feature_list, train_drug_A,train_drug_B,train_type,test_drug_A,test_drug_B,test_type,valid_drug_A,valid_drug_B,valid_type,train_smile_A,train_smile_B,test_smile_A, test_smile_B,valid_smile_A, valid_smile_B):

    vector = np.zeros((len(np.array(df_drug['drug_id']).tolist()), 0), dtype=float)  # vector=[]
    d_feature = {}
    for i in feature_list:
        # vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
        tempvec = feature_vector(i, df_drug)
        vector = np.hstack((vector, tempvec))
    # Transfrom the drug ID to feature vector
    for i in range(len(np.array(df_drug['drug_id']).tolist())):
        d_feature[np.array(df_drug['drug_id']).tolist()[i]] = vector[i]

    # Use the dictionary to obtain feature vector and label
    new_feature_train = []
    new_feature_test = []
    new_feature_valid = []
    new_label_train = []
    new_label_test = []
    new_label_valid = []

    for i in range(len(train_type)):
        temp = np.hstack((d_feature[train_drug_A[i]], d_feature[train_drug_B[i]]))
        new_feature_train.append(temp)
        new_label_train.append(train_type[i])

    for i in range(len(test_type)):
        temp = np.hstack((d_feature[test_drug_A[i]], d_feature[test_drug_B[i]]))
        new_feature_test.append(temp)
        new_label_test.append(test_type[i])

    for i in range(len(valid_type)):
        temp = np.hstack((d_feature[valid_drug_A[i]], d_feature[valid_drug_B[i]]))
        new_feature_valid.append(temp)
        new_label_valid.append(valid_type[i])

    # new_feature = np.array(new_feature)
    # new_label = np.array(new_label)
    new_feature_train = np.array(new_feature_train)
    new_feature_test = np.array(new_feature_test)
    new_feature_valid = np.array(new_feature_valid)
    new_label_train = np.array(new_label_train)
    new_label_test = np.array(new_label_test)
    new_label_valid = np.array(new_label_valid)

    print('new_feature_train_shape:',new_feature_train.shape)
    print('new_feature_test_shape:', new_feature_test.shape)
    print('new_feature_test_shape:', new_feature_valid.shape)
    print('new_label_train_shape:',new_label_train.shape)
    print('new_label_test_shape:', new_label_test.shape)
    print('new_label_valid_shape:', new_label_valid.shape)

    # tri_list = [(h, t, r) for h, t, r in zip(drugA, drugB, type)]
    tri_list = [(h, t, r) for h, t, r in zip(train_drug_A, train_drug_B, train_type)]
    tri_list_copy = []
    for h, t, r, *_ in tri_list:
        if ((h in MOL_EDGE_LIST_FEAT_MTX) and (t in MOL_EDGE_LIST_FEAT_MTX)):
            tri_list_copy.append((h, t, r))
    d1, d2, *_ = zip(*tri_list_copy)
    drug_ids = np.array(list(set(d1 + d2)))
    # print(len(drug_ids))
    drug_ids = np.array([id for id in drug_ids if id in MOL_EDGE_LIST_FEAT_MTX])
    # print(len(drug_ids))
    pos_h_samples = []
    pos_t_samples = []
    for h, t, _ in tri_list:
        h_data = create_graph_data(h)
        t_data = create_graph_data(t)
        pos_h_samples.append(h_data)
        pos_t_samples.append(t_data)
    train_h = pos_h_samples
    train_t = pos_t_samples

    tri_list = [(h, t, r) for h, t, r in zip(test_drug_A, test_drug_B, test_type)]
    tri_list_copy = []
    for h, t, r, *_ in tri_list:
        if ((h in MOL_EDGE_LIST_FEAT_MTX) and (t in MOL_EDGE_LIST_FEAT_MTX)):
            tri_list_copy.append((h, t, r))
    d1, d2, *_ = zip(*tri_list_copy)
    drug_ids = np.array(list(set(d1 + d2)))
    # print(len(drug_ids))
    drug_ids = np.array([id for id in drug_ids if id in MOL_EDGE_LIST_FEAT_MTX])
    # print(len(drug_ids))
    pos_h_samples = []
    pos_t_samples = []
    for h, t, _ in tri_list:
        h_data = create_graph_data(h)
        t_data = create_graph_data(t)
        pos_h_samples.append(h_data)
        pos_t_samples.append(t_data)
    test_h = pos_h_samples
    test_t = pos_t_samples


    tri_list = [(h, t, r) for h, t, r in zip(valid_drug_A, valid_drug_B, valid_type)]
    tri_list_copy = []
    for h, t, r, *_ in tri_list:
        if ((h in MOL_EDGE_LIST_FEAT_MTX) and (t in MOL_EDGE_LIST_FEAT_MTX)):
            tri_list_copy.append((h, t, r))
    d1, d2, *_ = zip(*tri_list_copy)
    drug_ids = np.array(list(set(d1 + d2)))
    # print(len(drug_ids))
    drug_ids = np.array([id for id in drug_ids if id in MOL_EDGE_LIST_FEAT_MTX])
    # print(len(drug_ids))
    pos_h_samples = []
    pos_t_samples = []
    for h, t, _ in tri_list:
        h_data = create_graph_data(h)
        t_data = create_graph_data(t)
        pos_h_samples.append(h_data)
        pos_t_samples.append(t_data)
    valid_h = pos_h_samples
    valid_t = pos_t_samples


    data = pd.read_csv(r'./data/train_fra.csv')
    data = np.array(data)
    train_fra_feature = data[:, 1:]
    print('train_fra_feature_shape:',train_fra_feature.shape)

    data = pd.read_csv(r'./data/test_fra.csv')
    data = np.array(data)
    test_fra_feature = data[:, 1:]
    print('test_fra_feature_shape:', test_fra_feature.shape)

    data = pd.read_csv(r'./data/valid_fra.csv')
    data = np.array(data)
    valid_fra_feature = data[:, 1:]
    print('valid_fra_feature_shape:', valid_fra_feature.shape)

    event_num = 2

    return new_feature_train,new_feature_test,new_feature_valid,new_label_train,new_label_test,new_label_valid,train_h,train_t,test_h,test_t,valid_h,valid_t,train_fra_feature,test_fra_feature,valid_fra_feature,event_num

    # return new_feature, new_label, pos_h_samples, pos_t_samples, drugA, drugB, event_num, fra_feature

def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)

        numerator = matrix * matrix.T

        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(sim_matrix[0])))
    return sim_matrix

MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol) for drug_id, mol in drug_id_mol_graph_tup}
MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}
TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])


class DDIDataset(Dataset):
    def __init__(self, x, nodeA, nodeAe, nodeB, nodeBe, y, x_fra):
        self.x = x
        self.nodeA = nodeA
        self.nodeAe = nodeAe
        self.nodeB = nodeB
        self.nodeBe = nodeBe
        self.y = y
        self.x_fra = x_fra

    def __getitem__(self, index):
        return self.x[index], self.nodeA[index], self.nodeAe[index], self.nodeB[index], self.nodeBe[index], self.y[index], self.x_fra[index]

    def __len__(self):
        return len(self.x)

    def collate_fn(self, batch):

        x_batch = np.zeros((1, 3096))
        x_fra_batch = np.zeros((1,1722))
        y_batch = np.zeros((1, 1))
        pos_h_samples = []
        pos_t_samples = []

        for x, nodeA, nodeAe, nodeB, nodeBe, y, x_fra in batch:
            x = x[np.newaxis,:]
            x_fra = x_fra[np.newaxis,:]
            x_batch = np.concatenate((x_batch, x), axis=0)
            x_fra_batch = np.concatenate((x_fra_batch, x_fra), axis=0)
            y = np.array([y])
            y = y[np.newaxis, :]
            y_batch = np.concatenate((y_batch, y), axis=0)
            nodeA = torch.FloatTensor(nodeA)
            nodeB = torch.FloatTensor(nodeB)
            nodeAe = torch.LongTensor(nodeAe)
            nodeBe = torch.LongTensor(nodeBe)
            pos_h_samples.append(Data(node=nodeA, edge_index=nodeAe))
            pos_t_samples.append(Data(node=nodeB, edge_index=nodeBe))


        y_batch = np.squeeze(y_batch)
        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_tri = (pos_h_samples, pos_t_samples)
        x_batch = torch.LongTensor(x_batch[1:])
        x_fra_batch = torch.LongTensor(x_fra_batch[1:])
        y_batch = torch.LongTensor(y_batch[1:])
        return x_batch, y_batch, pos_tri, x_fra_batch#nodeA_batch[1:], nodeAe_batch[:, 1:], nodeB_batch[1:], nodeBe_batch[:, 1:]

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

df_drug = pd.read_csv('./data/AMDE_drug.csv')
train_data = pd.read_csv('./data/train.csv')
train_drug_A, train_drug_B, train_type = train_data['d1'], train_data['d2'], train_data['type']
train_smile_A, train_smile_B = train_data['smile1'], train_data['smile2']
test_data = pd.read_csv('./data/test.csv')
test_drug_A, test_drug_B, test_type = test_data['d1'], test_data['d2'], test_data['type']
test_smile_A, test_smile_B = test_data['smile1'], test_data['smile2']
valid_data = pd.read_csv('./data/valid.csv')
valid_drug_A, valid_drug_B, valid_type = valid_data['d1'], valid_data['d2'], valid_data['type']
valid_smile_A, valid_smile_B = valid_data['smile1'], valid_data['smile2']
new_feature_train,new_feature_test,new_feature_valid,new_label_train,new_label_test,new_label_valid,train_h,train_t,test_h,test_t,valid_h,valid_t,train_fra_feature,test_fra_feature,valid_fra_feature,event_num = prepare(df_drug,feature_list=["smile"],train_drug_A=train_drug_A,train_drug_B=train_drug_B,train_type=train_type,test_drug_A=test_drug_A,test_drug_B=test_drug_B,test_type=test_type,
                                                                                         valid_drug_A=valid_drug_A,valid_drug_B=valid_drug_B,valid_type=valid_type,train_smile_A=train_smile_A,
                                                                                         train_smile_B=train_smile_B,test_smile_A=test_smile_A, test_smile_B=test_smile_B,valid_smile_A=valid_smile_A, valid_smile_B=valid_smile_B)
new_feature_train = new_feature_train.astype(np.longlong)
new_feature_test = new_feature_test.astype(np.longlong)
new_feature_valid = new_feature_valid.astype(np.longlong)
new_label_train = new_label_train.astype(np.longlong)
new_label_test = new_label_test.astype(np.longlong)
new_label_valid = new_label_valid.astype(np.longlong)
train_fra_feature = train_fra_feature.astype(np.longlong)
test_fra_feature = test_fra_feature.astype(np.longlong)
valid_fra_feature = valid_fra_feature.astype(np.longlong)


node_A = []
node_A_list = []
node_B = []
node_B_list = []
node_A_test = []
node_A_list_test = []
node_B_test = []
node_B_list_test = []
for i in tqdm(range(len(train_h))):
    node_A.append(list(train_h[i][0]))
    node_B.append(list(train_t[i][0]))
    node_A_list.append(list(train_h[i][1]))
    node_B_list.append(list(train_t[i][1]))
for j in tqdm(range(len(test_h))):
    node_A_test.append(list(test_h[j][0]))
    node_B_test.append(list(test_t[j][0]))
    node_A_list_test.append(list(test_h[j][1]))
    node_B_list_test.append(list(test_t[j][1]))

n_atom_feats = TOTAL_ATOM_FEATS
n_atom_hid = 64

model = My_Model(len(new_feature_train[0]), 3, 4, 2, n_atom_feats, n_atom_hid,
                     heads_out_feat_params=[32, 32, 32, 32], blocks_params=[4, 4, 4, 4], fra_dim=len(train_fra_feature[0]))


x_test = new_feature_test
nodeA_feature_test = node_A_test
nodeA_edg_test = node_A_list_test
nodeB_feature_test = node_B_test
nodeB_edg_test = node_B_list_test
y_test = new_label_test
Xfra_test = test_fra_feature
test_dataset = DDIDataset(x_test, nodeA_feature_test, nodeA_edg_test, nodeB_feature_test, nodeB_edg_test,
                          np.array(y_test), Xfra_test)
test_loader = DrugDataLoader(data=test_dataset, batch_size=128, shuffle=False)


PATH = 'state_dict_model_1.pth'
model.load_state_dict(torch.load(PATH))
model.eval()

y_pre_all_test = []
y_true_all_test = []

X_vis = np.zeros((1,2560))
with torch.no_grad():
    for batch_idx, data_test in enumerate(test_loader, 0):
        x_batch_test, y_batch_test, a_test, x_fra = data_test
        x_batch_test = x_batch_test
        y_true_test = y_batch_test
        x_fra = x_fra
        a_test = [tensor for tensor in a_test]
        y_pre_test,vis = model(x_batch_test.float(), a_test, x_fra.float())
        X_vis= np.concatenate((X_vis, vis.cpu().numpy()), axis=0)
        y_pre_all_test.append(F.softmax(y_pre_test).cpu().numpy())
        y_true_all_test.append(y_true_test.cpu().numpy())

y_pre_all_test_score = np.concatenate(y_pre_all_test)
y_true_all_test = np.concatenate(y_true_all_test)
y_pre_all_test = np.argmax(y_pre_all_test_score, axis=1)




vis = X_vis[1:]

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt




def plot_embedding_3d(X, y, title=None):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i, 2], str(y[i]),
                color=plt.cm.Set3(y[i] / 5.),
                fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)
    #plt.show()
    plt.savefig('E://achemso/4.eps')


print("Computing t-SNE embedding")
tsne3d = TSNE(n_components=3, init='pca', random_state=0)

X_tsne_3d = tsne3d.fit_transform(vis)
plot_embedding_3d(X_tsne_3d[:, 0:3], y_true_all_test, "DDI")






