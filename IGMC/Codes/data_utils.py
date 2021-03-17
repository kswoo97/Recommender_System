import numpy as np
import random

def map_data(data) :

    uniq = list(set(data))
    id_dict = {old : new for new, old in enumerate(sorted(uniq))}
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)

    return data, id_dict, n

def load_data(dataset) :

    """

    :param dataset: 그냥 csv 형태의 데이터를 넣어주면 됩니다. 주의할 점은 UserID / Movie ID / Rating / Timestamp 순입니다.
    :return:
    ---
    num_users : int / Returns number of Users
    num_items : int / Returns number of Items
    u_nodes : array / 출력되는 User 번호입니다.
    v_nodes : array / 출력되는 Item 번호입니다.
    ratings : Rating을 출력합니다.
    u_features : 피쳐를 출력하는데 딱히 필요해보이지는 않습니다.
    v_features : 이 역시 필요없어요.
    """
    u_features = None
    v_features = None

    data_array = dataset.values

    u_nodes_ratings = data_array[:, 0]
    v_nodes_ratings = data_array[:, 1]
    ratings = data_array[:, 2]

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
    ratings = ratings.astype(np.float32)

    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features, u_dict, v_dict