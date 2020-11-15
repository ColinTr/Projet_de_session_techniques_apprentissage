from sklearn.model_selection import StratifiedShuffleSplit


def sampling(x_train, t_train):
    sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

    for train_index, test_index in sss:
        X_train, X_test = train.values[train_index], train.values[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
    return


def error():
    return