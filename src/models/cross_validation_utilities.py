from sklearn.model_selection import StratifiedShuffleSplit


def sampling(x_data, t_data, k_fold=10, test_size=0.2):

    sss = StratifiedShuffleSplit(n_splits=k_fold, test_size=test_size)
    for train_index, test_index in sss.split(x_data, t_data):
        x_train, x_validation = x_data[train_index], x_data[test_index]
        t_train, t_validation = t_data[train_index], t_data[test_index]
    return


def error():
    return
