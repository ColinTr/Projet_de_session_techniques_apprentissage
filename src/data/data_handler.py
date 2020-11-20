# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def center_data(data):
    return data - data.mean()


def normalize_data_by_standard_deviation(data):
    return data.div(data.std(axis=0, ddof=0), axis=1)


def normalize_data_by_min_max(data):
    return data.div(data.std(axis=0, ddof=0), axis=1)


def parse_csv_file(filepath):
    return pd.read_csv(filepath)


class DataHandler:
    def __init__(self, train_fp, output_fp):
        self.data_input_filepath = train_fp
        self.output_filepath = output_fp
        self.output_files_paths = []

        self.species_column = None
        self.data = None
        self.data_normalized_centered = None
        self.labels = None
        self.species = None

    def main(self):
        """ Runs data processing scripts to turn raw data from (../raw) into
            cleaned data ready to be analyzed (saved in ../processed).
        """

        self.data = parse_csv_file(self.data_input_filepath)

        self.encode_species(self.data)

        data_centered = center_data(self.data)
        self.data_normalized_centered = normalize_data_by_standard_deviation(data_centered)

        self.export_data_into_csv()

    def encode_species(self, data_to_encode):
        self.species_column = data_to_encode.species
        le = LabelEncoder().fit(data_to_encode.species)
        self.labels = le.transform(data_to_encode.species)
        self.species = list(le.classes_)

        self.data = data_to_encode.drop(['species', 'id'], axis=1)
        return

    def read_all_output_files(self):
        data = parse_csv_file(self.output_files_paths[0]).to_numpy()
        cn_data = parse_csv_file(self.output_files_paths[1]).to_numpy()
        labels = parse_csv_file(self.output_files_paths[2]).to_numpy()
        species = parse_csv_file(self.output_files_paths[3]).to_numpy()
        return data, cn_data, labels, species

    def export_data_into_csv(self):
        # We lookup the actual file names
        filename = os.path.basename(self.data_input_filepath)

        # We create the export files paths
        data_fp = self.output_filepath + '/data-processed-' + filename
        centered_normalized_data_fp = \
            self.output_filepath + '/data-centered-normalized-processed-' + filename
        labels_fp = self.output_filepath + '/labels-processed-' + filename
        species_fp = self.output_filepath + '/species-processed-' + filename

        # We save the files paths for future read
        self.output_files_paths.append(data_fp)
        self.output_files_paths.append(centered_normalized_data_fp)
        self.output_files_paths.append(labels_fp)
        self.output_files_paths.append(species_fp)

        # We export our data
        self.data.to_csv(data_fp, index=False)
        self.data_normalized_centered.to_csv(centered_normalized_data_fp, index=False)
        pd.DataFrame(data=self.labels, columns=["label_num"]).to_csv(labels_fp, index=False)
        pd.DataFrame(data=self.species_column, columns=["species"]).to_csv(species_fp, index=False)
        return


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            "Usage: python data_handler.py train_data_input_filepath output_filepath\n")
        print(
            "Exemple (Windows) : python src\\data\\data_handler.py data\\raw\\train\\leaf-classification-train.csv "
            "data\\processed\n")
        print("Exemple (Linux) : python src/data/data_handler.py data/raw/train/leaf-classification-train.csv "
              "data/processed\n")
    else:
        train_data_input_filepath = sys.argv[1]
        output_filepath = sys.argv[2]

        DataHandler(train_data_input_filepath, output_filepath).main()
