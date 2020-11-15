# -*- coding: utf-8 -*-
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    if len(sys.argv) < 4:
        print("Usage: python gestion_donnees.py test_data_input_filepath train_data_input_filepath output_filepath\n")
        print("Exemple (Windows) : python src\\data\\gestion_donnees.py data\\raw\\test\\leaf-classification-test.csv "
              "data\\raw\\train\\leaf-classification-train.csv data\\processed\n")
        print("Exemple (Linux) : python src/data/gestion_donnees.py data/raw/test/leaf-classification-test.csv "
              "data/raw/train/leaf-classification-train.csv data/processed\n")
        return

    test_data_input_filepath = sys.argv[1]
    train_data_input_filepath = sys.argv[2]
    output_filepath = sys.argv[3]

    train_data, test_data = parser_donnees_csv(train_data_input_filepath, test_data_input_filepath)

    train_data, labels, test_data, test_ids, classes = encoder_especes(train_data, test_data)


def parser_donnees_csv(train_data_input_filepath, test_data_input_filepath):
    train_data = pd.read_csv(train_data_input_filepath)
    test_data = pd.read_csv(test_data_input_filepath)
    return train_data, test_data


def encoder_especes(training_data, testing_data):
    le = LabelEncoder().fit(training_data.species)
    labels = le.transform(training_data.species)
    classes = list(le.classes_)
    test_ids = testing_data.id

    training_data = training_data.drop(['species', 'id'], axis=1)
    testing_data = testing_data.drop(['id'], axis=1)

    return training_data, labels, testing_data, test_ids, classes


def centrer_donnees():
    return


def normaliser_donnees():
    return


def exporter_donnees_en_csv():
    return


if __name__ == '__main__':
    main()
