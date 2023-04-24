import os
from collections import defaultdict
from tqdm import tqdm
import json

import numpy as np
import pandas as pd

import nibabel as nib
import SimpleITK as sitk
from SimpleITK import GetImageFromArray
from radiomics.featureextractor import RadiomicsFeatureExtractor
import albumentations as A

import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
 
feature_names = [
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

texture_extractor = RadiomicsFeatureExtractor(verbose=False)


def get_paths(json_path, seg_path):
    t1, t1ce, t2, flair, mgmt_label = [], [], [], [], []
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    
    for i in json_dict["training"]:
        t1.append("../data/" + i["image"][0])
        t1ce.append("../data/" + i["image"][1])
        t2.append("../data/" + i["image"][2])
        flair.append("../data/" + i["image"][3])
        mgmt_label.append(i["mgmt_label"])

    id_ls = [i.split("/")[-2].split("_")[-1] for i in t1]
    seg = [os.path.join(seg_path,i+".nii.gz") for i in id_ls]
    id_ls = [int(i) for i in id_ls]
    
    df = pd.DataFrame.from_dict({
        "id" : id_ls,
        "BraTS21ID" : id_ls,
        "t1" : t1,
        "t1ce" : t1ce,
        "t2" : t2,
        "flair" : flair,
        "seg" : seg,
    })
    df.set_index("id", inplace=True)

    return df, mgmt_label


class ImageReader:
    def __init__(self, img_size=256, normalize=False, single_class=False, class_ = "multi"):
        pad_size = 256 if img_size > 256 else 224 # 96
        self.resize = A.Compose(
            [
                A.PadIfNeeded(min_height=pad_size, min_width=pad_size, value=0),
                A.Resize(img_size, img_size)
            ]
        )
        self.normalize=normalize
        self.single_class=single_class
        self.class_ = class_
        
    def read_file(self, path, mask_path) :
        raw_image = nib.load(path).get_fdata()
        raw_mask = nib.load(mask_path).get_fdata()
        processed_frames, processed_masks = [], []
        for frame_idx in range(raw_image.shape[2]):
            frame = raw_image[:, :, frame_idx]
            mask = raw_mask[:, :, frame_idx]
            if self.normalize:
                if frame.max() > 0:
                    frame = frame/frame.max()
                frame = frame.astype(np.float32)
            else:
                frame = frame.astype(np.uint8)
            resized = self.resize(image=frame, mask=mask)
            processed_frames.append(resized['image'])
            
            if self.class_ == "multi":
                processed_masks.append(1*(resized['mask'] > 0))
            elif self.class_ == 1:
                processed_masks.append(1*(resized['mask'] == 1))
            elif self.class_ == 2:
                processed_masks.append(1*(resized['mask'] == 2))
            else:
                processed_masks.append(1*(resized['mask'] == 4))
        return {
            'scan': np.stack(processed_frames, 0),
            'segmentation': np.stack(processed_masks, 0),
            'orig_shape': raw_image.shape
        }
    
    def load_patient_scan(self, df, idx, modality = 'flair') :
        return self.read_file(df.loc[idx][modality], df.loc[idx]["seg"])
    


def get_radiomice_features(df, idx, reader, modality="flair"):
    data = reader.load_patient_scan(df, idx, modality)
    scan = sitk.GetImageFromArray(data["scan"])
    mask = sitk.GetImageFromArray(data["segmentation"])
    features = texture_extractor.execute(scan,mask)
    tmp_df = pd.DataFrame([features]).T
    row = [idx]
    col_names = ["BraTS21ID"]
    
    for col in feature_names[1:]:
        if "BoundingBox" in col:
            row.extend(list(tmp_df.loc[col][0]))
            for j in range(6):
                col_names.append(col+"_"+str(j))
        elif "CenterOfMassIndex" in col:
            row.extend(list(tmp_df.loc[col][0]))
            for j in range(3):
                col_names.append(col+"_"+str(j))
        else:
            row.append(tmp_df.loc[col][0])
            col_names.append(col)
        
    return row, col_names



def get_features(json_path, seg_path):
    df, mgmt_labels = get_paths(json_path, seg_path)
    id_ls = list(df["BraTS21ID"])

    features = defaultdict(lambda : defaultdict(dict))
    reader = ImageReader(img_size=128, normalize=True, single_class=False, class_=1)

    for modality in ["t1", "t2", "t1ce", "flair"]:
        for class_ in ["multi", 1, 2, 4]:
            reader = ImageReader(img_size=128, normalize=True, single_class=False, class_=class_)
            for i in tqdm(id_ls):
                try:
                    tmp_features, column_names = get_radiomice_features(df, i,reader,modality) 
                    features[modality][class_][i] = tmp_features
                except Exception as e:
                    print(e)

    return features, mgmt_labels, column_names


