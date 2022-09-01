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

    # for i in range(len(type)):
    #     temp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))
    #     new_feature.append(temp)
    #     new_label.append(type[i])
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

    # item = torch.zeros((1,1722))
    # for i in tqdm(range(len(train_smile_A))):
    #     fuse = smiles2vector(train_smile_A[i],train_smile_B[i])
    #     fuse = np.reshape(fuse, (1, -1))
    #     item = np.vstack((item,fuse))
    # train_fra_feature = item[1:,:]
    #
    # item = torch.zeros((1, 1722))
    # for i in tqdm(range(len(test_smile_A))):
    #     fuse = smiles2vector(test_smile_A[i], test_smile_B[i])
    #     fuse = np.reshape(fuse, (1, -1))
    #     item = np.vstack((item, fuse))
    # test_fra_feature = item[1:, :]
    #
    # item = torch.zeros((1, 1722))
    # for i in tqdm(range(len(valid_smile_A))):
    #     fuse = smiles2vector(valid_smile_A[i], valid_smile_B[i])
    #     fuse = np.reshape(fuse, (1, -1))
    #     item = np.vstack((item, fuse))
    # valid_fra_feature = item[1:, :]

    # print('train_fra:',train_fra_feature.shape)
    # print('test_fra:', test_fra_feature.shape)
    # print('valid_fra:',valid_fra_feature.shape)
    # res = DataFrame(train_fra_feature)
    # res.to_csv(r'C:\Users\Administrator\Desktop\train_fra.csv')
    # res = DataFrame(test_fra_feature)
    # res.to_csv(r'C:\Users\Administrator\Desktop\test_fra.csv')
    # res = DataFrame(valid_fra_feature)
    # res.to_csv(r'C:\Users\Administrator\Desktop\valid_fra.csv')

    data = pd.read_csv(r'./data/train_fra.csv')
    data = np.array(data)
    train_fra_feature = data[:, 1:]
    print('train_fra_feature_shape:',train_fra_feature.shape)
    #
    data = pd.read_csv(r'./data/test_fra.csv')
    data = np.array(data)
    test_fra_feature = data[:, 1:]
    print('test_fra_feature_shape:', test_fra_feature.shape)
    #
    data = pd.read_csv(r'./data/valid_fra.csv')
    data = np.array(data)
    valid_fra_feature = data[:, 1:]
    print('valid_fra_feature_shape:', valid_fra_feature.shape)

    # pos_h_samples = Batch.from_data_list(pos_h_samples)
    # pos_t_samples = Batch.from_data_list(pos_t_samples)
    # pos_tri = (pos_h_samples, pos_t_samples)
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

#..

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
        #nodeA_batch = []
        #nodeAe_batch = []
        #nodeB_batch = []
        #nodeBe_batch = []
        x_batch = np.zeros((1, 3096))
        x_fra_batch = np.zeros((1,1722))
        y_batch = np.zeros((1, 1))
        pos_h_samples = []
        pos_t_samples = []
        #nodeA_batch = np.zeros((1, 55))
        #nodeB_batch = np.zeros((1, 55))
        #nodeAe_batch = np.zeros((2,1))
        #nodeBe_batch = np.zeros((2,1))
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
            #nodeA_batch = np.concatenate((nodeA_batch, nodeA), axis=0)
            #nodeB_batch = np.concatenate((nodeB_batch, nodeB), axis=0)
            #nodeAe_batch = np.concatenate((nodeAe_batch, nodeAe), axis=1)
            #nodeBe_batch = np.concatenate((nodeBe_batch, nodeBe), axis=1)

        y_batch = np.squeeze(y_batch)
        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_tri = (pos_h_samples, pos_t_samples)
        #x_batch = torch.from_numpy(x_batch[1:])
        x_batch = torch.LongTensor(x_batch[1:])
        x_fra_batch = torch.LongTensor(x_fra_batch[1:])
        #y_batch = torch.from_numpy(y_batch[1:])
        y_batch = torch.LongTensor(y_batch[1:])
        return x_batch, y_batch, pos_tri, x_fra_batch#nodeA_batch[1:], nodeAe_batch[:, 1:], nodeB_batch[1:], nodeBe_batch[:, 1:]

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)








