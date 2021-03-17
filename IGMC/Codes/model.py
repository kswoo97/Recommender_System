import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, RGCNConv, global_sort_pool, global_add_pool
from torch_geometric.utils import dropout_adj
from util_functions import *
import pdb
import time

class GNN (torch.nn.Module) :

    # 중요한 GNN 관련 dropout이나 기초 Layer를 쌓아줍니다.

    def __init__ (self, dataset, gconv = GCNConv, latent_dim = [32, 32, 32, 1], regression = False,
                  adj_dropout = 0.2, force_undirected = False) :
        super(GNN, self).__init__()
        self.regression = regression
        self.adj_dropout = adj_dropout
        self.force_undirected = force_undirected
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0]))
        for i in range(0, len(latent_dim) - 1) :
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1]))
        self.lin1 = Linear(sum(latent_dim), 128)
        if self.regression :
            self.lin2 = Linear(128, 1)
        else :
            self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self) :
        for conv in self.convs :
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward (self, data) :
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0 :
            edge_index, edge_type, dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training = self.training
            )
        concat_states = []
        for conv in self.convs :
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_add_pool(concat_states, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.lin2(x)
        if self.regression :
            return x[:, 0]
        else :
            return F.log_softmax(x, dim = -1)

    def __repr__ (self) :
        return self.__class__.__name__

class IGMC(GNN) :
    # 우리가 최종적으로 구현하고자 하는 핵심 모델!
    # Inductive Graph Based Matrix Completion
    def __init__ (self, dataset, gconv = RGCNConv, latent_dim = [32,32,32,32],
                  num_relations = 5, num_bases = 2, regression = False, adj_dropout = 0.2,
                  force_undirected = False, side_features = False, n_side_features = 0,
                  multiply_by = 1) :
        """
        :param dataset: 일반적인 데이터를 넣어주면 됩니다.
        :param gconv: 어떤 GCN을 사용할 것인지 말해줍니다. RGCNConv를 인자로 놓아야 합니다.
        :param latent_dim: 각 Layer의 차원을 정의합니다.
        :param num_relations: Edge의 Type이 몇 개 존재하는지 확인합니다.
        :param num_bases: Regularization으로 선택할 Bases의 수입니다. 많을수록 자세한 학습 But Overfitting
        :param regression: Task를 Regression으로 할거나 Classification으로 할거냐를 정의합니다.
        :param adj_dropout: 오피비팅을 방지하기 위해 그래프에 대한 Edge Dropout을 적용합니다
        :param force_undirected: 방향을 정의하는데, False로 놓으면 됩니다.
        :param side_features: 기타 정보가 있는 경우 입력합니다.
        :param n_side_features: 모델에 제공할 기본 정보의 수입니다.
        :param multiply_by: Decay 값인데, 대부분 1로 설정합니다.
        """
        super(IGMC, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected)
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases)) # 입력에 대한 첫 Layer
        for i in range(0, len(latent_dim) - 1) :
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases)) # 순서대로 Stacking
        self.lin1 = Linear(2*sum(latent_dim), 128)
        self.lin3 = Linear(64, 32)
        self.side_features = side_features
        if side_features :
            self.lin1 = Linear(2*sum(latent_dim) + n_side_features, 128)

    def forward(self, data):
        start = time.time()
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs :
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        # Target User와 Item에 해당하는 Hidden Values만
        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features :
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        # Feature Extraction은 종료. 원하는 Value도 뽑았다.

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        #x = F.relu(self.lin3(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression :
            return x[: , 0] * self.multiply_by
        else :
            return F.log_softmax(x, dim = -1)

class IGMC2(GNN) :
    # 우리가 최종적으로 구현하고자 하는 핵심 모델!
    # Inductive Graph Based Matrix Completion
    def __init__ (self, dataset, gconv = RGCNConv, latent_dim = [32,32,32,32],
                  num_relations = 5, num_bases = 5, regression = False, adj_dropout = 0.2,
                  force_undirected = False, side_features = False, n_side_features = 0,
                  multiply_by = 1) :
        """
        :param dataset: 일반적인 데이터를 넣어주면 됩니다.
        :param gconv: 어떤 GCN을 사용할 것인지 말해줍니다. RGCNConv를 인자로 놓아야 합니다.
        :param latent_dim: 각 Layer의 차원을 정의합니다.
        :param num_relations: Edge의 Type이 몇 개 존재하는지 확인합니다.
        :param num_bases: Regularization으로 선택할 Bases의 수입니다. 많을수록 자세한 학습 But Overfitting
        :param regression: Task를 Regression으로 할거나 Classification으로 할거냐를 정의합니다.
        :param adj_dropout: 오피비팅을 방지하기 위해 그래프에 대한 Edge Dropout을 적용합니다
        :param force_undirected: 방향을 정의하는데, False로 놓으면 됩니다.
        :param side_features: 기타 정보가 있는 경우 입력합니다.
        :param n_side_features: 모델에 제공할 기본 정보의 수입니다.
        :param multiply_by: Decay 값인데, 대부분 1로 설정합니다.
        """
        super(IGMC, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected)
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases)) # 입력에 대한 첫 Layer
        for i in range(0, len(latent_dim) - 1) :
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases)) # 순서대로 Stacking
        self.lin1 = Linear(2*sum(latent_dim), 128)
        self.side_features = side_features
        if side_features :
            self.lin1 = Linear(2*sum(latent_dim) + n_side_features, 128)

    def forward(self, data):
        start = time.time()
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs :
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        # Target User와 Item에 해당하는 Hidden Values만
        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features :
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        # Feature Extraction은 종료. 원하는 Value도 뽑았다.

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression :
            return x[: , 0] * self.multiply_by
        else :
            return F.log_softmax(x, dim = -1)

