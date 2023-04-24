import os
import numpy as np
import pandas as pd

from extraction import get_features
from preprocessing import clean_features, preprocess_df, feature_selection
from training import train_model
from utils import save_json, save_npy, load_json, load_npy
from ensemble import get_feature_array


def main():
    # os.system("python ../test.py --data_dir ../data --json_list ../jsons/metadata.json --feature_size 12 --roi_x=96 --roi_y=96 --roi_z=96 --pretrained_dir ../runs/alpha_0.9_enc3")
    train_json_path = "../jsons/train_metadata.json"
    test_json_path = "../jsons/test_metadata.json"
    seg_path = "../runs/alpha_0.9_enc1"
    feature_vector = False
    feature_select = False

    print("Extracting Radiomics Features")
    if os.path.exists(os.path.join(seg_path, "train_radiomics.json")):
        train_features = load_json(os.path.join(seg_path, "train_radiomics.json"))
        train_labels = load_npy(os.path.join(seg_path, "train_labels.npy"))
        test_features = load_json(os.path.join(seg_path, "test_radiomics.json"))
        test_labels = load_npy(os.path.join(seg_path, "test_labels.npy"))
    else:
        train_features, train_labels, _ = get_features(train_json_path, os.path.join(seg_path, "outputs"))
        test_features, test_labels, _ = get_features(test_json_path, os.path.join(seg_path, "outputs"))

        save_json(train_features, os.path.join(seg_path, "train_radiomics.json"))
        save_npy(np.array(train_labels), os.path.join(seg_path, "train_labels.npy"))
        save_json(test_features, os.path.join(seg_path, "test_radiomics.json"))
        save_npy(np.array(test_labels), os.path.join(seg_path, "test_labels.npy"))

    print("Cleaning Features")
    train_df, train_labels, train_ids  = clean_features(train_features, train_labels)
    test_df, test_labels, test_ids = clean_features(test_features, test_labels)

    id_col = list(train_df.columns[0:1])
    train_cols = list(train_df.columns[1:])
    test_cols = list(test_df.columns[1:])
    cont_cols = list(set(train_cols) & set(test_cols))
    
    train_df = train_df[id_col + cont_cols]
    test_df = test_df[id_col + cont_cols]
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    encoders = {}
    na_vals_dict = {}

    print("Preprocessing DataFrame")
    train_df, encoders, na_vals_dict = preprocess_df(train_df, [], cont_cols, encoders, na_vals_dict, data = "train")
    test_df, encoders, na_vals_dict = preprocess_df(test_df, [], cont_cols, encoders, na_vals_dict, data = "test")

    print("Performing Feature Selection")
    if feature_select:
        train_ds, test_ds = feature_selection(train_df, test_df, train_labels, type="frufs")
    train_ds, test_ds = train_df, test_df

    if feature_vector:
        train_seg_feature_arr, train_pt_ids = get_feature_array(train_json_path, os.path.join(seg_path, "outputs"))
        test_seg_feature_arr, test_pt_ids = get_feature_array(test_json_path, os.path.join(seg_path, "outputs"))

        train_seg_feature_arr = np.concatenate([train_seg_feature_arr[i] for i in train_ids])
        test_seg_feature_arr = np.concatenate([test_seg_feature_arr[i] for i in test_ids])
        train_pt_ids = [train_pt_ids[i] for i in train_ids]
        test_pt_ids = [test_pt_ids[i] for i in test_ids]

        train_seg_feature_df = pd.DataFrame(train_seg_feature_arr, columns = [f"Seg_{i}" for i in len(train_seg_feature_arr)])
        train_seg_feature_df["BraTS21ID"] = train_pt_ids
        test_seg_feature_df = pd.DataFrame(test_seg_feature_arr, columns = [f"Seg_{i}" for i in len(test_seg_feature_arr)])
        test_seg_feature_df["BraTS21ID"] = test_pt_ids

        train_ds = train_ds.merge(train_seg_feature_df, on="BraTS21ID")
        test_ds = test_ds.merge(test_seg_feature_df, on="BraTS21ID")

    print("Training Model")
    model = train_model(train_ds, train_labels, test_ds, test_labels)


if __name__ == "__main__":
    main()