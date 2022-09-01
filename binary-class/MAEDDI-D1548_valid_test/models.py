import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.container import ModuleList
import math
from torch_geometric.nn import (GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_mean_pool,
                                max_pool_neighbor_x,
                                global_add_pool)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


bert_n_heads = 3
drop_out_rating = 0.3
len_after_AE = 512

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X

class CrossAttention(nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(CrossAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.W_Q1 = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K1 = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V1 = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)
        self.fc1 = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X, X1):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        Q1 = self.W_Q1(X1).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K1 = self.W_K1(X1).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V1 = self.W_V1(X1).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q1, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores1 = torch.matmul(Q, K1.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        attn1 = torch.nn.Softmax(dim=-1)(scores1)
        context = torch.matmul(attn, V)
        context1 = torch.matmul(attn1, V1)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        context1 = context1.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        output1 = self.fc1(context1)
        return output, output1

class EncoderLayertwo(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayertwo, self).__init__()
        self.attn = CrossAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

        self.AN1_1 = torch.nn.LayerNorm(input_dim)

        self.l1_1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2_1 = torch.nn.LayerNorm(input_dim)

    def forward(self, X, X1):
        output, output1 = self.attn(X, X1)

        X = self.AN1(output + X)
        output = self.l1(X)
        X = self.AN2(output + X)

        X1 = self.AN1_1(output1 + X1)
        output1 = self.l1_1(X1)
        X1 = self.AN2_1(output1 + X1)

        return X,X1

class EN1(torch.nn.Module):
    def __init__(self, vector_size):
        super(EN1, self).__init__()

        self.vector_size = vector_size

        self.l1 = torch.nn.Linear(self.vector_size, self.vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(self.vector_size // 2)
        self.att1 = EncoderLayer(self.vector_size // 2, bert_n_heads)

        self.l2 = torch.nn.Linear(self.vector_size // 2, len_after_AE*2)
        self.bn2 = torch.nn.BatchNorm1d(len_after_AE*2)
        self.att2 = EncoderLayer(len_after_AE*2, bert_n_heads)

        self.l3 = torch.nn.Linear(len_after_AE*2, len_after_AE)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.att1(X)

        X = self.dr(self.bn2(self.ac(self.l2(X))))
        X = self.att2(X)

        X = self.l3(X)

        return X

class EN2(torch.nn.Module):
    def __init__(self, vector_size):
        super(EN2, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(vector_size // 2, len_after_AE*2)
        self.bn1 = torch.nn.BatchNorm1d(len_after_AE*2)
        self.att1 = EncoderLayer(len_after_AE*2, bert_n_heads)

        self.l2 = torch.nn.Linear(vector_size // 2, len_after_AE * 2)
        self.bn2 = torch.nn.BatchNorm1d(len_after_AE * 2)
        self.att2 = EncoderLayer(len_after_AE * 2, bert_n_heads)

        self.l3 = torch.nn.Linear(len_after_AE * 2, len_after_AE)
        self.bn3 = torch.nn.BatchNorm1d(len_after_AE)

        self.l4 = torch.nn.Linear(len_after_AE * 2, len_after_AE)
        self.bn4 = torch.nn.BatchNorm1d(len_after_AE)

        self.att3 = EncoderLayertwo(len_after_AE, bert_n_heads)

        self.l5 = torch.nn.Linear(len_after_AE * 2, len_after_AE)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]

        X1 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X1 = self.att1(X1)

        X2 = self.dr(self.bn2(self.ac(self.l2(X2))))
        X2 = self.att2(X2)

        X1 = self.dr(self.bn3(self.ac(self.l3(X1))))
        X2 = self.dr(self.bn4(self.ac(self.l4(X2))))

        X1, X2 = self.att3(X1, X2)

        X = torch.cat((X1, X2),dim=1)

        X = self.l5(X)

        return X








class SSI_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, data):
        data.node = self.conv(data.node, data.edge_index)
        #att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores = self.readout(node, nodee,
                                                                                             #batch=torch.LongTensor(np.array([100])))
        #att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores = self.readout(node, nodee,)
        att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores = self.readout(data.node, data.edge_index,
                                                                                             batch=data.batch)
        global_graph_emb = global_add_pool(att_x, att_batch)

        # data = max_pool_neighbor_x(data)
        return data, global_graph_emb

class FraEN(torch.nn.Module):
    def __init__(self, input_dim):
        super(FraEN, self).__init__()

        self.l1 = torch.nn.Linear(input_dim, len_after_AE*2)
        self.bn1 = torch.nn.BatchNorm1d(len_after_AE*2)
        self.attn1 = EncoderLayer(len_after_AE*2, bert_n_heads)

        self.l2 = torch.nn.Linear(len_after_AE*2,len_after_AE+200)
        self.bn2 = torch.nn.BatchNorm1d(len_after_AE+200)
        self.attn2 = EncoderLayer(len_after_AE+200,bert_n_heads)

        self.l3 = torch.nn.Linear(len_after_AE+200,len_after_AE)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.attn1(X)

        X = self.dr(self.bn2(self.ac(self.l2(X))))
        X = self.attn2(X)

        X = self.l3(X)

        return X



class My_Model(nn.Module):
    def __init__(self, input_dim, n_heads, n_layers, event_num,in_features, hidd_dim, heads_out_feat_params, blocks_params, fra_dim):
        super(My_Model, self).__init__()
        self.en1 = EN1(input_dim)
        self.en2 = EN2(input_dim)
        self.fraEN = FraEN(fra_dim)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.input_dim = input_dim

        self.layers = torch.nn.ModuleList([EncoderLayer(len_after_AE * 5, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(len_after_AE * 5)

        self.l1 = torch.nn.Linear(len_after_AE * 5, len_after_AE)
        self.bn1 = torch.nn.BatchNorm1d(len_after_AE)

        self.l2 = torch.nn.Linear(len_after_AE, len_after_AE//4)
        self.bn2 = torch.nn.BatchNorm1d(len_after_AE//4)

        self.l3 = torch.nn.Linear(len_after_AE//4, len_after_AE // 8)
        self.bn3 = torch.nn.BatchNorm1d(len_after_AE // 8)

        self.l4 = torch.nn.Linear(len_after_AE // 8, event_num)

        self.ac = gelu




        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.n_blocks = len(blocks_params)

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()

        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = SSI_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        self.attn1 = EncoderLayer(len_after_AE // 4, bert_n_heads)
        self.attn2 = EncoderLayer(len_after_AE // 4, bert_n_heads)
        self.attn3 = EncoderLayer(len_after_AE // 4, bert_n_heads)
        self.attn4 = EncoderLayer(len_after_AE // 4, bert_n_heads)
        self.attns = []
        self.attns.append(self.attn1)
        self.attns.append(self.attn2)
        self.attns.append(self.attn3)
        self.attns.append(self.attn4)

        self.attn5 = EncoderLayer(len_after_AE // 4, bert_n_heads)
        self.attn6 = EncoderLayer(len_after_AE // 4, bert_n_heads)
        self.attn7 = EncoderLayer(len_after_AE // 4, bert_n_heads)
        self.attn8 = EncoderLayer(len_after_AE // 4, bert_n_heads)
        self.attns1 = []
        self.attns1.append(self.attn5)
        self.attns1.append(self.attn6)
        self.attns1.append(self.attn7)
        self.attns1.append(self.attn8)

        self.l5 = torch.nn.Linear(len_after_AE, len_after_AE // 2)
        self.bn5 = torch.nn.BatchNorm1d(len_after_AE // 2)

        self.l6 = torch.nn.Linear(len_after_AE, len_after_AE // 2)
        self.bn6 = torch.nn.BatchNorm1d(len_after_AE // 2)

        self.att_cross = EncoderLayertwo(len_after_AE // 2, bert_n_heads)


    def forward(self, X, triples, X_fra):
        X1 = self.en1(X)
        X2 = self.en2(X)
        X3 = self.fraEN(X_fra)




        h_data, t_data = triples
        h_data.node = self.initial_norm(h_data.node, h_data.batch)
        t_data.node = self.initial_norm(t_data.node, t_data.batch)

        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out1, out2 = block(h_data), block(t_data)
            h_data = out1[0]
            t_data = out2[0]
            r_h = out1[1]
            r_t = out2[1]
            r_h = self.attns[i](r_h)
            r_t = self.attns1[i](r_t)
            repr_h.append(r_h)
            repr_t.append(r_t)
            h_data.node = F.elu(self.net_norms[i](h_data.node, h_data.batch))
            t_data.node = F.elu(self.net_norms[i](t_data.node, t_data.batch))

        repr_h = torch.cat(repr_h, dim=1)
        repr_h = self.dr(self.bn5(self.ac(self.l5(repr_h))))
        repr_t = torch.cat(repr_t, dim=1)
        repr_t = self.dr(self.bn6(self.ac(self.l6(repr_t))))

        repr_h, repr_t = self.att_cross(repr_h, repr_t)

        X4 = torch.cat((repr_h, repr_t), dim=1)

        X5 = X1 + X2 + X3 + X4
        X = torch.cat((X1,X2,X3,X4,X5),dim=1)

        for layer in self.layers:
            X = layer(X)
        X_vis = self.AN(X)

        X = self.dr(self.bn1(self.ac(self.l1(X_vis))))
        X = self.dr(self.bn2(self.ac(self.l2(X))))
        X = self.dr(self.bn3(self.ac(self.l3(X))))

        X = self.l4(X)

        return X, X_vis
