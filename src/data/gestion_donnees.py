# -*- coding: utf-8 -*-
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class GestionnaireDonnees:
    def __init__(self, train_data_input_filepath, test_data_input_filepath, output_filepath):
        self.train_data_input_filepath = train_data_input_filepath
        self.test_data_input_filepath = test_data_input_filepath
        self.output_filepath = output_filepath

        self.train_data = None
        self.test_data = None

        self.labels = None
        self.test_ids = None
        self.classes = None

    def main(self):
        """ Runs data processing scripts to turn raw data from (../raw) into
            cleaned data ready to be analyzed (saved in ../processed).
        """

        self.parser_donnees_csv(self.train_data_input_filepath, self.test_data_input_filepath)

        self.encoder_especes(self.train_data, self.test_data)

    def parser_donnees_csv(self, train_filepath, test_filepath):
        self.train_data = pd.read_csv(train_filepath)
        self.test_data = pd.read_csv(test_filepath)
        return

    def encoder_especes(self, training_data, testing_data):
        le = LabelEncoder().fit(training_data.species)
        self.labels = le.transform(training_data.species)
        self.classes = list(le.classes_)
        self.test_ids = testing_data.id

        self.train_data = training_data.drop(['species', 'id'], axis=1)
        self.test_data = testing_data.drop(['id'], axis=1)
        return

    def centrer_donnees(self):
        return

    def normaliser_donnees(self):
        return

    def exporter_donnees_en_csv(self):
        return


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(
            "Usage: python gestion_donnees.py test_data_input_filepath train_data_input_filepath output_filepath\n")
        print(
            "Exemple (Windows) : python src\\data\\gestion_donnees.py data\\raw\\test\\leaf-classification-test.csv "
            "data\\raw\\train\\leaf-classification-train.csv data\\processed\n")
        print("Exemple (Linux) : python src/data/gestion_donnees.py data/raw/train/leaf-classification-train.csv "
              "data/raw/test/leaf-classification-test.csv data/processed\n")
    else:
        train_data_input_filepath = sys.argv[1]
        test_data_input_filepath = sys.argv[2]
        output_filepath = sys.argv[3]
        GestionnaireDonnees(train_data_input_filepath, test_data_input_filepath, output_filepath).main()