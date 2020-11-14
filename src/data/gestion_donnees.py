# -*- coding: utf-8 -*-
import sys
import pandas as pd


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    if len(sys.argv) < 4:
        print("Usage: python gestion_donnees.py test_data_input_filepath train_data_input_filepath output_filepath\n")
        print(" exemple: python3 regression.py 1 sin 20 20 0.3 10 0.001\n")
        return

    test_data_input_filepath = sys.argv[1]
    train_data_input_filepath = sys.argv[2]
    output_filepath = sys.argv[3]

    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

if __name__ == '__main__':
    main()