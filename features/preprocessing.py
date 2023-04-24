from tqdm import tqdm

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from FRUFS import FRUFS
from PyImpetus import PPIMBC
from sklearn.feature_selection import chi2, SelectKBest, f_classif
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from lightgbm import LGBMClassifier
import lightgbm as lgb

column_names = [
    "BraTS21ID",\
    "diagnostics_Mask-original_BoundingBox", \
    "diagnostics_Mask-original_CenterOfMassIndex",\
    "diagnostics_Image-original_Mean", \
    "original_shape_Elongation",\
    "original_shape_Flatness", \
    "original_shape_LeastAxisLength",\
    "original_shape_MajorAxisLength",\
    "original_shape_Maximum2DDiameterColumn",\
    "original_shape_Maximum2DDiameterRow",\
    "original_shape_Maximum2DDiameterSlice",\
    "original_shape_Maximum3DDiameter",\
    "original_shape_MeshVolume",\
    "original_shape_MinorAxisLength",\
    "original_shape_Sphericity",\
    "original_shape_SurfaceArea",\
    "original_shape_SurfaceVolumeRatio",\
    "original_firstorder_10Percentile",\
    "original_firstorder_90Percentile",\
    "original_firstorder_Energy",\
    "original_firstorder_InterquartileRange",\
    "original_firstorder_Kurtosis",\
    "original_firstorder_Maximum",\
    "original_firstorder_MeanAbsoluteDeviation",\
    "original_firstorder_Mean",\
    "original_firstorder_Median",\
    "original_firstorder_Minimum",\
    "original_firstorder_Range",\
    "original_firstorder_RobustMeanAbsoluteDeviation",\
    "original_firstorder_RootMeanSquared",\
    "original_firstorder_Skewness",\
    "original_firstorder_TotalEnergy",\
    "original_firstorder_Variance",\
    "original_gldm_DependenceEntropy",\
    "original_gldm_DependenceNonUniformity",\
    "original_gldm_DependenceNonUniformityNormalized",\
    "original_gldm_DependenceVariance",\
    "original_gldm_GrayLevelNonUniformity",\
    "original_gldm_LargeDependenceEmphasis",\
    "original_gldm_LargeDependenceHighGrayLevelEmphasis",\
    "original_gldm_LargeDependenceLowGrayLevelEmphasis",\
    "original_gldm_SmallDependenceEmphasis",\
    "original_gldm_SmallDependenceHighGrayLevelEmphasis",\
    "original_gldm_SmallDependenceLowGrayLevelEmphasis",\
    "original_glrlm_GrayLevelNonUniformity",\
    "original_glrlm_LongRunEmphasis",\
    "original_glrlm_LongRunHighGrayLevelEmphasis",\
    "original_glrlm_LongRunLowGrayLevelEmphasis",\
    "original_glrlm_RunEntropy",\
    "original_glrlm_RunLengthNonUniformity",\
    "original_glrlm_RunLengthNonUniformityNormalized",\
    "original_glrlm_RunPercentage",\
    "original_glrlm_RunVariance",\
    "original_glrlm_ShortRunEmphasis",\
    "original_glrlm_ShortRunHighGrayLevelEmphasis",\
    "original_glrlm_ShortRunLowGrayLevelEmphasis",\
    "original_glszm_GrayLevelNonUniformity",\
    "original_glszm_HighGrayLevelZoneEmphasis",\
    "original_glszm_LargeAreaEmphasis",\
    "original_glszm_LargeAreaLowGrayLevelEmphasis",\
    "original_glszm_SizeZoneNonUniformity",\
    "original_glszm_SizeZoneNonUniformityNormalized",\
    "original_glszm_SmallAreaEmphasis",\
    "original_glszm_SmallAreaHighGrayLevelEmphasis",\
    "original_glszm_SmallAreaLowGrayLevelEmphasis",\
    "original_glszm_ZoneEntropy",\
    "original_glszm_ZonePercentage",\
    "original_glszm_ZoneVariance",\
]



def clean_features(features, labels):
    id_ls = list(features["t1"]["multi"].keys())
    rad_features = []  
    new_labels = []
    ids = []
    for i, id_ in tqdm(enumerate(id_ls)):
        tmp_features = [id_]
        try:
            for modality in ["t1", "t2", "t1ce", "flair"]:
                for seg in ["multi", "1", "2", "4"]:
                    tmp_features.extend(features[modality][seg][id_][1:])
            rad_features.append(tmp_features)
            new_labels.append(labels[i])
            ids.append(i)
        except Exception as e:
            # print(e)
            continue
    
    final_col_names = ["BraTS21ID"]
    final_col_names.extend("Feature_"+str(i+1) for i in range(len(rad_features[-1])-1))

    # for modality in ["t1", "t2", "t1ce", "flair"]:
    #     for seg in ["multi", "1", "2", "4"]:
    #         for col in column_names[1:]:
    #             final_col_names.append("_".join([modality,seg,col]))
    
    rad_df = pd.DataFrame(rad_features)
    rad_df.columns = final_col_names
    rad_df["label"] = new_labels
    
    for col in list(rad_df.columns[:-1]):
        dtype = rad_df[col].dtype
        if dtype == "object":
            rad_df[col] = rad_df[col].astype(float)

    rad_df = rad_df.T.drop_duplicates().T
    rad_df = rad_df.sort_values(by="BraTS21ID")
    
    rad_df["BraTS21ID"] = rad_df["BraTS21ID"].astype(int)

    labels = rad_df["label"]
    rad_df.drop(["label"], axis=1, inplace=True)
    return rad_df, labels, ids


def preprocess_df(df, cat_features, cont_features, encoders, na_vals_dict, data = "train"):
    for col in cat_features:
        if data == "train":
            na_val = df[col].value_counts().idxmax()
            df[col].fillna(na_val,inplace=True)
            cat_encoder = LabelEncoder()
            df[col] = cat_encoder.fit_transform(np.array(df[col]).reshape(-1, 1))
            encoders[col] = cat_encoder
            na_vals_dict[col] = na_val
        else:
            df[col].fillna(na_vals_dict[col],inplace=True)
            df[col] = encoders[col].transform(np.array(df[col]).reshape(-1, 1))
    
            
    for col in cont_features: 
        if data == "train":
            na_val = np.mean(df[col])
            df[col].fillna(na_val,inplace=True)
            cont_encoder = MinMaxScaler() #StandardScaler()
            df[col] = cont_encoder.fit_transform(np.array(df[col]).reshape(-1, 1))
            encoders[col] = cont_encoder
            na_vals_dict[col] = na_val
        else:
            df[col].fillna(na_vals_dict[col],inplace=True)
            df[col] = encoders[col].transform(np.array(df[col]).reshape(-1, 1))

    return df, encoders, na_vals_dict


def feature_selection(train_ds, test_ds, train_labels, type="frufs"):
    if type == "frufs":
        model_frufs_generated = FRUFS(
            model_r=lgb.LGBMRegressor(random_state=42),
            k=15
        )
        train_ds = model_frufs_generated.fit_transform(train_ds)
        test_ds = model_frufs_generated.transform(test_ds)

    elif type == "pca":
        pca = PCA(n_components=150)

        train_ds = pca.fit_transform(train_ds)
        test_ds = pca.transform(test_ds)

        train_ds = pd.DataFrame(train_ds)
        test_ds = pd.DataFrame(test_ds)

    elif type == "chi2":
        feature_cols = list(train_ds.columns)
        kbest = SelectKBest(chi2, k=15) #chi2 f_classif
        train_ds = kbest.fit_transform(train_ds, train_labels)
        test_ds = kbest.transform(test_ds)

        cols = kbest.get_feature_names_out(feature_cols)

        train_ds = pd.DataFrame(train_ds, columns=cols)
        test_ds = pd.DataFrame(test_ds, columns=cols)

    elif type == "pyimpetus":
        model = LGBMClassifier(
                       n_estimators = 1000,
                       metric = "auc"
            )
        pyimp = PPIMBC(model=model, p_val_thresh=0.05, num_simul=5, simul_size=0.2, simul_type=0, sig_test_type="non-parametric", cv=5, random_state=2023, n_jobs=-1, verbose=2)
        train_ds = pyimp.fit_transform(train_ds, train_labels)
        test_ds = pyimp.transform(test_ds)

    return train_ds, test_ds


def get_preprocess_df(train_df, test_df, encoders, na_vals_dict):
    feature_cols = list(train_df.columns)[1:-1]

    train_df = preprocess_df(train_df, [], feature_cols, encoders, na_vals_dict, data = "train")
    test_df = preprocess_df(test_df, [], feature_cols, encoders, na_vals_dict, data = "test")

    train_labels = np.array(train_df["MGMT_value"])
    train_ds = train_df.drop(["MGMT_value", "BraTS21ID"], axis = 1).reset_index(drop=True)

    test_labels = np.array(test_df["MGMT_value"])
    test_ds = test_df.drop(["MGMT_value", "BraTS21ID"], axis = 1).reset_index(drop=True)

    train_ds, test_ds = feature_selection(train_ds, test_ds, train_labels)

    return train_ds, train_labels, test_ds, test_labels


