# Morph-DeepSets

A lot of this code is taken from [this github from Patrick Odagiu et all](https://github.com/fastmachinelearning/l1-jet-id/tree/main)

The Preprocessing folder (specifically the equalize_normalise function) use the raw_data folder to make a k-fold normalized dataset. 
for example run python equalise_normalise.py --x_data_path_train ../raw_data/train_processed/x_jet_images_c8_minpt2.npy --y_data_path_train ../raw_data/train_processed/y_jet_images_c8_minpt2.npy --x_data_path_test ../raw_data/val_processed/x_jet_images_c8_minpt2.npy --y_data_path_test ../raw_data/val_processed/y_jet_images_c8_minpt2.npy --feats allfeats --norm robust_fast --shuffle_seed 41 --kfolds 5 --output_dir ../normalized_data

For the argument norm you have the options of ["nonorm", "standard", "robust", "robust_fast", "minmax"], the default is "robust_fast"

To download the data visit this [google drive link](https://drive.google.com/drive/folders/1ZqxwL8A5RuWYPEH5pzvUSOzjI3uqLE9E?usp=drive_link)

