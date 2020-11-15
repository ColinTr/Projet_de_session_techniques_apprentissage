import sys

from src.data.data_handler import DataHandler
from src.data.data_handler import parse_csv_file


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python data_handler.py train_data_input_filepath output_filepath\n")
        print(
            "Exemple (Windows) : python main.py data\\raw\\train\\leaf-classification-train.csv data\\processed\n")
        print("Exemple (Linux) : python main.py data/raw/train/leaf-classification-train.csv data/processed\n")

    else:
        data_input_filepath = sys.argv[1]
        output_filepath = sys.argv[2]

        print("=============== Reading and handling data ===============")
        dh = DataHandler(data_input_filepath, output_filepath)
        dh.main()
        data, data_normalized_centered, labels, species =\
            dh.read_all_output_files()

        print(data)
        print(data_normalized_centered)
        print(labels)
        print(species)

        print("Done")
    return


if __name__ == '__main__':
    main()
