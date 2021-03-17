import numpy as np
import pandas as pd
import scipy.sparse as sp
from data_utils import load_data, map_data

def create_trainvaltest_split(dataset, validation = False, testing = False) :
    """
    Dataset should be given by csv file
    :param dataset: input CSV file should be given
    :return : 
    """
    None

    # 전에 만들어놨던 함수에서 한번에 불러옵니다.
    num_users, num_items, u_nodes, v_nodes, ratings, u_f, v_f, u_dict, v_dict = load_data(dataset)
    rating_dict = {r : i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    if validation :

        # Train / Test에 대한 Size를 정의합니다.
        print("Split dataset into train/val/test by time ...")
        num_train = int(ratings.shape[0] * 0.7)
        num_val = int(ratings.shape[0] * 0.8) - num_train
        num_test = ratings.shape[0] - num_train - num_val

        pairs_nonzero = np.vstack([u_nodes, v_nodes]).transpose()
        train_pairs_idx = pairs_nonzero[0:int(num_train)]
        val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
        test_pairs_idx = pairs_nonzero[num_train + num_val:]

        # 위에 애들인 서로 이미 데이터가 섞여있다는 가정이 있음을 주의해야 합니다.
        # 따라서 Shuffling을 필요에 따라 진행할수도 있습니다.

        # Index 단위로 이를 정렬합니다.
        u_test_idx, v_test_idx = test_pairs_idx.transpose()
        u_val_idx, v_val_idx = val_pairs_idx.transpose()
        u_train_idx, v_train_idx = train_pairs_idx.transpose()

        # 라벨을 생성합니다.
        all_labels = np.array([rating_dict[r] for r in ratings], dtype=np.int32)
        train_labels = all_labels[0:int(num_train)]
        val_labels = all_labels[num_train:num_train + num_val]
        test_labels = all_labels[num_train + num_val:]

        if testing:
            u_train_idx = np.hstack([u_train_idx, u_val_idx])
            v_train_idx = np.hstack([v_train_idx, v_val_idx])
            train_labels = np.hstack([train_labels, val_labels])

        class_values = np.sort(np.unique(ratings))

        data = train_labels + 1.

        # Sparse Matrix를 생성합니다.

        #rating_mx_train = sp.csr_matrix((data, [u_train_idx, v_train_idx]),
        #                                   shape=[num_users, num_items], dtype=np.float32)

        data2 = ratings + 1.
        csr_matrix = sp.csr_matrix((data2, [u_nodes, v_nodes]),
                                   shape = [num_users, num_items], dtype = np.float32)
        # 원래는 아래있는 애로
        return csr_matrix, train_labels, u_train_idx, v_train_idx, val_labels, u_val_idx, v_val_idx, \
               test_labels, u_test_idx, v_test_idx, class_values

    else :

        # Train / Test에 대한 Size를 정의합니다.
        print("Split dataset into train/val/test by time ...")
        num_train = int(ratings.shape[0] * 0.7)
        num_test = ratings.shape[0] - num_train

        pairs_nonzero = np.vstack([u_nodes, v_nodes]).transpose()
        train_pairs_idx = pairs_nonzero[0:int(num_train)]
        test_pairs_idx = pairs_nonzero[num_train:]

        # 위에 애들인 서로 이미 데이터가 섞여있다는 가정이 있음을 주의해야 합니다.
        # 따라서 Shuffling을 필요에 따라 진행할수도 있습니다.

        # Index 단위로 이를 정렬합니다.
        u_test_idx, v_test_idx = test_pairs_idx.transpose()
        u_train_idx, v_train_idx = train_pairs_idx.transpose()

        # 라벨을 생성합니다.
        all_labels = np.array([rating_dict[r] for r in ratings], dtype=np.int32)
        train_labels = all_labels[0:int(num_train)]
        test_labels = all_labels[num_train:]

        class_values = np.sort(np.unique(ratings))

        data = train_labels + 1.

        # Sparse Matrix를 생성합니다.

        rating_mx_train = sp.csr_matrix((data, [u_train_idx, v_train_idx]),
                                        shape=[num_users, num_items], dtype=np.float32)
        # 원래는 아래있는 애로
        return rating_mx_train, train_labels, u_train_idx, v_train_idx, \
               test_labels, u_test_idx, v_test_idx, class_values,

#


