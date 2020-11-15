# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def center_data(data):
    return data.sub(data.mean(axis=0), axis=1)


def normalize_data(data):
    return data.div(data.std(axis=0), axis=1)


def parse_csv_file(filepath):
    return pd.read_csv(filepath)


class DataHandler:
    def __init__(self, train_fp, test_fp, output_fp):
        self.train_data_input_filepath = train_fp
        self.test_data_input_filepath = test_fp
        self.output_filepath = output_fp
        self.output_files_paths = []

        self.train_data = None
        self.test_data = None
        self.train_data_normalized_centered = None

        self.labels = None
        self.test_ids = None
        self.species = None

    def main(self):
        """ Runs data processing scripts to turn raw data from (../raw) into
            cleaned data ready to be analyzed (saved in ../processed).
        """

        self.train_data = parse_csv_file(self.train_data_input_filepath)
        self.test_data = parse_csv_file(self.test_data_input_filepath)

        self.encode_species(self.train_data, self.test_data)

        train_data_centered = center_data(self.train_data)
        self.train_data_normalized_centered = normalize_data(train_data_centered)

        self.export_data_into_csv()

    def encode_species(self, training_data, testing_data):
        le = LabelEncoder().fit(training_data.species)
        self.labels = le.transform(training_data.species)
        self.species = list(le.classes_)

        self.test_ids = testing_data.id
        self.train_data = training_data.drop(['species', 'id'], axis=1)
        self.test_data = testing_data.drop(['id'], axis=1)
        return

    def read_all_output_files(self):
        trd = parse_csv_file(self.output_files_paths[0])
        cntd = parse_csv_file(self.output_files_paths[1])
        ted = parse_csv_file(self.output_files_paths[2])
        tl = parse_csv_file(self.output_files_paths[3])
        ts = parse_csv_file(self.output_files_paths[4])
        ti = parse_csv_file(self.output_files_paths[4])
        return trd, cntd, ted, tl, ts, ti

    def export_data_into_csv(self):
        # We lookup the actual file names
        train_fn = os.path.basename(self.train_data_input_filepath)
        test_fn = os.path.basename(self.test_data_input_filepath)

        # We create the export files paths
        train_data_fp = self.output_filepath + '/train-data-processed-' + train_fn
        centered_normalized_train_data_fp =\
            self.output_filepath + '/train-data-centered-normalized-processed-' + train_fn
        test_data_fp = self.output_filepath + '/test-data-processed-' + test_fn
        train_labels_fp = self.output_filepath + '/train-labels-processed-' + train_fn
        train_species_fp = self.output_filepath + '/train-species-processed-' + train_fn
        test_ids_fp = self.output_filepath + '/test-ids-processed-' + test_fn

        # We save the files paths for future read
        self.output_files_paths.append(train_data_fp)
        self.output_files_paths.append(centered_normalized_train_data_fp)
        self.output_files_paths.append(test_data_fp)
        self.output_files_paths.append(train_labels_fp)
        self.output_files_paths.append(train_species_fp)
        self.output_files_paths.append(test_ids_fp)

        # We export our data
        self.train_data.to_csv(train_data_fp, index=False)
        self.train_data_normalized_centered.to_csv(centered_normalized_train_data_fp, index=False)
        self.test_data.to_csv(test_data_fp, index=False)
        pd.DataFrame(data=self.labels, columns=["label_num"]).to_csv(train_labels_fp, index=False)
        pd.DataFrame(data=self.species, columns=["species"]).to_csv(train_species_fp, index=False)
        pd.DataFrame(data=self.test_ids, columns=["id"]).to_csv(test_ids_fp, index=False)
        return


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(
            "Usage: python data_handler.py test_data_input_filepath train_data_input_filepath output_filepath\n")
        print(
            "Exemple (Windows) : python src\\data\\data_handler.py data\\raw\\train\\leaf-classification-train.csv "
            "data\\raw\\test\\leaf-classification-test.csv data\\processed\n")
        print("Exemple (Linux) : python src/data/data_handler.py data/raw/train/leaf-classification-train.csv "
              "data/raw/test/leaf-classification-test.csv data/processed\n")
    else:
        train_data_input_filepath = sys.argv[1]
        test_data_input_filepath = sys.argv[2]
        output_filepath = sys.argv[3]

        DataHandler(train_data_input_filepath, test_data_input_filepath, output_filepath).main()
