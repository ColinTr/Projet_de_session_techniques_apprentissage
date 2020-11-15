from sklearn.model_selection import StratifiedShuffleSplit


def sampling(x_train, t_train, k_fold=10, test_size=0.2):

    sss = StratifiedShuffleSplit(k_fold, test_size)
    for train_index, test_index in sss.split(x_train, t_train):
        X_train, X_test = x_train[train_index], x_train[test_index]
        y_train, y_test = t_train[train_index], t_train[test_index]
    return


def error():
    return