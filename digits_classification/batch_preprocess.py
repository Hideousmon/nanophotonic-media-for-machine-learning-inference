import numpy as np

def generate_index_pool(data):
    index_pool = np.array(range(0, np.shape(data)[0]))
    return index_pool

def random_select_from_index_pool(index_pool, batch_size):
    selected_indexes = []
    selecting_size = batch_size if batch_size < index_pool.size else index_pool.size
    for i in range(0, selecting_size):
        selected_index = np.random.randint(0, index_pool.size)
        selected_indexes.append(index_pool[selected_index])
        index_pool = np.delete(index_pool, selected_index)
    return index_pool, selected_indexes

def get_batch_data(train_data_features, train_data_targets, batch_size):
    batch_features_list = []
    batch_targets_list = []
    index_pool = generate_index_pool(train_data_features)
    while (index_pool.size != 0):
        index_pool, selected_indexes = random_select_from_index_pool(index_pool, batch_size=batch_size)
        features_list = []
        targets_list = []
        for index in selected_indexes:
            features_list.append(train_data_features[index])
            targets_list.append(train_data_targets[index])
        batch_features_list.append(np.array(features_list))
        batch_targets_list.append(np.array(targets_list))

    return batch_features_list, batch_targets_list
