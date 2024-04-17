# Normalise numpy arrays and split them into training, validation, and testing sub
# data sets to make them ready for the machine learning algorithms.
# Example command:
# python equalise_normalise.py --x_data_path_train ../raw_data/train_processed/x_jet_images_c8_minpt2.npy --y_data_path_train ../raw_data/train_processed/y_jet_images_c8_minpt2.npy --x_data_path_test ../raw_data/val_processed/x_jet_images_c8_minpt2.npy --y_data_path_test ../raw_data/val_processed/y_jet_images_c8_minpt2.npy --feats allfeats --norm robust_fast --shuffle_seed 41 --kfolds 5 --output_dir /Users/dimademler/Documents/GitHub/Morph-DeepSets/normalized_data

import os
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold

import feature_selection
import standardisation
# import plots
import util
from terminal_colors import tcols


def main(args):
    print("Loading the files...\n")
    seed = args.set_seed
    np.random.seed(seed)
    x_data_train = np.load(args.x_data_path_train, "r")
    y_data_train = np.load(args.y_data_path_train, "r")
    x_data_test = np.load(args.x_data_path_test, "r")
    y_data_test = np.load(args.y_data_path_test, "r")

    # Equalise the number of jets per class.
    x_data_train, y_data_train = util.equalise_classes(x_data_train, y_data_train)
    x_data_test, y_data_test = util.equalise_classes(x_data_test, y_data_test)

    # Perform feature selection.
    x_data_train = feature_selection.get_features_numpy(x_data_train, args.feats)
    x_data_test = feature_selection.get_features_numpy(x_data_test, args.feats)

    print("Normalising data...")
    norm_params = standardisation.fit_standardisation(args.norm, x_data_train)
    x_data_train = standardisation.apply_standardisation(
        args.norm, x_data_train, norm_params
    )
    x_data_test = standardisation.apply_standardisation(
        args.norm, x_data_test, norm_params
    )

    # print("Plotting training data...")
    # plots_folder = format_output_filename(args.x_data_path_train, args.feats, args.norm)
    # plots_path = os.path.join(args.output_dir, plots_folder)
    # plots.constituent_number(plots_path, x_data_train)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # train_folder = os.path.basename(os.path.dirname(args.x_data_path_train))
    # test_folder = os.path.basename(os.path.dirname(args.x_data_path_test))
    base_name = os.path.splitext(os.path.basename(args.x_data_path_train))[0]

    # Create the output file names
    train_output_name = f"{base_name[2:]}_{args.feats}_{args.norm}"
    print("train_output_name: ", train_output_name)
    test_output_name = f"{base_name[2:]}_{args.feats}_{args.norm}"
    print("test_output_name: ", test_output_name)

    print(tcols.OKGREEN + "Saving preprocessed data..." + tcols.ENDC)
    np.save(os.path.join(args.output_dir, f"x_train_{train_output_name}.npy"), x_data_train)
    np.save(os.path.join(args.output_dir, f"y_train_{train_output_name}.npy"), y_data_train)
    np.save(os.path.join(args.output_dir, f"x_test_{test_output_name}.npy"), x_data_test)
    np.save(os.path.join(args.output_dir, f"y_test_{test_output_name}.npy"), y_data_test)

    util.save_hyperparameters_file(vars(args), args.output_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--x_data_path_train",
        type=str,
        required=True,
        help="Path to the training data file to process.",
    )
    parser.add_argument(
        "--y_data_path_train",
        type=str,
        required=True,
        help="Paths to the training target file corresponding to the data.",
    )
    parser.add_argument(
        "--x_data_path_test",
        type=str,
        required=True,
        help="Path to the test data file to process.",
    )
    parser.add_argument(
        "--y_data_path_test",
        type=str,
        required=True,
        help="Paths to the test target file corresponding to the data.",
    )
    parser.add_argument(
        "--feats",
        type=str,
        default="ptetaphi",
        choices=["ptetaphi", "allfeats"],
        help="The type of feature selection to be employed.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="nonorm",
        choices=["nonorm", "standard", "robust", "robust_fast", "minmax"],
        help="The type of normalisation to apply to the data.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output folder."
    )

    parser.add_argument(
        "--set_seed", type=int, default=42, help="Seed for shuffling the jets in the data set."
    )
    args = parser.parse_args()
    main(args)
