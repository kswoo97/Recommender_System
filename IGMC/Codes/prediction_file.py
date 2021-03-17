import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_utils import *
from preprocessing import *
from util_functions import *
from model import *
from train import *
from torch import torch
import multiprocessing as mp

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

class IGMC_rec_interface() :

    def __init__(self, data, movie_feauture):

        self.data = data
        self.movie_feature = movie_feauture
        (self.n_users, self.n_items, self.u_nodes, self.v_nodes, self.ratings,
         self.u_f, self.v_f, self.u_dict, self.v_dict) = load_data(data)
        self.movie_list = np.unique(data.movieId)
        csm = sp.csr_matrix((self.ratings, [self.u_nodes, self.v_nodes]),
                            shape=[self.n_users, self.n_items], dtype=np.float32)
        csc = csm.tocsc()
        self.csm = csm
        self.csc = csc
        self.initializer = test_data_generator((self.u_dict[1], self.v_dict[1]), csm, csc)

    def check_user_preference (self, user_number) :

        self.user = user_number
        movie_list = self.movie_feature[self.movie_feature['movieId'].isin(self.data[self.data.userId == user_number].movieId.values)]
        ratings = self.data[self.data.userId == user_number].sort_values('movieId').ratings.values
        movie_list['ratings'] = ratings
        return movie_list

    def recommend(self, user_num, n_rec, model) :

        csm = sp.csr_matrix((self.ratings, [self.u_nodes, self.v_nodes]),
                                        shape=[self.n_users, self.n_items], dtype=np.float32)
        csc = csm.tocsc()
        Arow = SparseRowIndexer(csm)
        Acol = SparseColIndexer(csc)
        watched_list = self.data[self.data.userId==user_num]['movieId'].values # 안 본 영화에 대해서만 예측 Rating

        # 매번 DataFrame에 접근해서 영화정보를 띄우는 것은 그다지 효율적이지 않음.
        # n_rec (추천받는 음악의 개수만큼만 계속 랭크 높은 순으로 최신화)
        candidate_list = np.linspace(0, 1, n_rec)
        movie_index = np.linspace(0, 1, n_rec)
        self.prediction = []

        with torch.no_grad() :

            for i in tqdm(self.movie_list) :

                if np.sum(watched_list==i) == 0 :  # 해당 영화를 안보았다면 Rating을 예측해라!

                    data = test_data_generator((self.u_dict[user_num], self.v_dict[i]), Arow, Acol) # Subgraph Input 생성
                    predict_rating = model(data.to(device))[0]

                    if np.min(candidate_list) < predict_rating :

                        cur_ind = np.where(candidate_list == np.min(candidate_list))[0][0]
                        candidate_list[cur_ind] = predict_rating
                        movie_index[cur_ind] = i

                    self.prediction.append(predict_rating)


            # 다 거치면 최종적으로 n_rec개만큼의 영화가 남게된다.
            print('추천이 끝났습니다! {}번 유저의 추천 결과물은 아래와 같습니다.'.format(user_num))
            self.output = candidate_list

        return self.movie_feature[self.movie_feature['movieId'].isin(movie_index)]