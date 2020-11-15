# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class GestionnaireDonnees:
    def __init__(self, train_fp, test_fp, output_fp):
        self.train_data_input_filepath = train_fp
        self.test_data_input_filepath = test_fp
        self.output_filepath = output_fp

        self.train_data = None
        self.test_data = None

        self.labels = None
        self.test_ids = None
        self.species = None

    def main(self):
        """ Runs data processing scripts to turn raw data from (../raw) into
            cleaned data ready to be analyzed (saved in ../processed).
        """

        self.parser_donnees_csv(self.train_data_input_filepath, self.test_data_input_filepath)

        self.encoder_especes(self.train_data, self.test_data)

        self.exporter_donnees_en_csv()

    def parser_donnees_csv(self, train_filepath, test_filepath):
        self.train_data = pd.read_csv(train_filepath)
        self.test_data = pd.read_csv(test_filepath)
        return

    def encoder_especes(self, training_data, testing_data):
        le = LabelEncoder().fit(training_data.species)
        self.labels = le.transform(training_data.species)
        self.species = list(le.classes_)

        self.test_ids = testing_data.id
        self.train_data = training_data.drop(['species', 'id'], axis=1)
        self.test_data = testing_data.drop(['id'], axis=1)
        return

    def centrer_donnees(self):
        return

    def normaliser_donnees(self):
        return

    def exporter_donnees_en_csv(self):
        # On cherche le nom des fichiers
        train_fn = os.path.basename(self.train_data_input_filepath)
        test_fn = os.path.basename(self.test_data_input_filepath)

        # On crée le nom des fichiers que l'on va exporter
        train_data_fp = self.output_filepath + '/train-data-processed-' + train_fn
        test_data_fp = self.output_filepath + '/test-data-processed-' + test_fn
        train_labels_fp = self.output_filepath + '/train-labels-processed-' + train_fn
        train_species_fp = self.output_filepath + '/train-species-processed-' + train_fn
        test_ids_fp = self.output_filepath + '/test-ids-processed-' + test_fn

        # On exporte nos données
        self.train_data.to_csv(train_data_fp, index=False)
        self.test_data.to_csv(test_data_fp, index=False)
        pd.DataFrame(data=self.labels, columns=["label_num"]).to_csv(train_labels_fp, index=False)
        pd.DataFrame(data=self.species, columns=["species"]).to_csv(train_species_fp, index=False)
        pd.DataFrame(data=self.test_ids, columns=["id"]).to_csv(test_ids_fp, index=False)
        return


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(
            "Usage: python gestion_donnees.py test_data_input_filepath train_data_input_filepath output_filepath\n")
        print(
            "Exemple (Windows) : python src\\data\\gestion_donnees.py data\\raw\\train\\leaf-classification-train.csv "
            "data\\raw\\test\\leaf-classification-test.csv data\\processed\n")
        print("Exemple (Linux) : python src/data/gestion_donnees.py data/raw/train/leaf-classification-train.csv "
              "data/raw/test/leaf-classification-test.csv data/processed\n")
    else:
        train_data_input_filepath = sys.argv[1]
        test_data_input_filepath = sys.argv[2]
        output_filepath = sys.argv[3]

        GestionnaireDonnees(train_data_input_filepath, test_data_input_filepath, output_filepath).main()
