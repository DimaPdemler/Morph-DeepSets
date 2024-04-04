# Morph-DeepSets

A lot of this code is taken from [this GitHub repo by Patrick Odagiu et al.](https://github.com/fastmachinelearning/l1-jet-id/tree/main)

## Preprocessing

The `Preprocessing` folder, specifically the `equalize_normalise` function, uses the `raw_data` folder to create a k-fold normalized dataset. For the `norm` argument, you have the following options: `["nonorm", "standard", "robust", "robust_fast", "minmax"]`. The default is `"robust_fast"`.

## Data Download

To download the data, visit this [Google Drive link](https://drive.google.com/drive/folders/1ZqxwL8A5RuWYPEH5pzvUSOzjI3uqLE9E?usp=drive_link).

## Model Training

If you want to recreate Patrick's model training, follow these steps:

1. Download the `raw_data` folder from the Google Drive and place it into Patrick's cloned repo at `./deepsets`.
2. Run the following command to train the model:

```
python equalise_normalise.py --x_data_path_train ../raw_data/train_processed/x_jet_images_c8_minpt2.npy --y_data_path_train ../raw_data/train_processed/y_jet_images_c8_minpt2.npy --x_data_path_test ../raw_data/val_processed/x_jet_images_c8_minpt2.npy --y_data_path_test ../raw_data/val_processed/y_jet_images_c8_minpt2.npy --feats allfeats --norm robust_fast --shuffle_seed 41 --kfolds 5 --output_dir ../normalized_data
```
