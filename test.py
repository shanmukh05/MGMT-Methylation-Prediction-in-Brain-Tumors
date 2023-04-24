# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from utils.data_utils import get_loader
from sklearn.metrics import roc_auc_score
from monai.transforms import Activations, AsDiscrete
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch

#from monai.inferers import sliding_window_inference
from utils.swi import sliding_window_inference
from utils.utils import AverageMeter, get_feature_vector
from model import SwinUNETR

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--data_dir", default="./data/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="./jsons/test_metadata.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=1, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="model_final.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument(
    "--pretrained_dir",
    default="./runs/test/",
    type=str,
    help="pretrained checkpoint directory",
)

def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = os.path.join(args.pretrained_dir, "outputs")
    if os.path.exists(output_directory):
        # shutil.rmtree(output_directory)
        # os.makedirs(output_directory)
        None
    else:
        os.makedirs(output_directory)

    test_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = SwinUNETR(
        img_size=128,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
        out_feature_vector=True
    )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    run_acc = AverageMeter()
    acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    dice_tc, dice_wt, dice_et = 0, 0, 0
    labels, preds, ids = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image, target, mgmt_target = batch["image"], batch["label"], batch["mgmt_label"]
            image, target, mgmt_target = image.cuda(), target.cuda(), mgmt_target.cuda()
            affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            
            num = batch["id"][0].split("/")[-1]
            img_name = num + ".nii.gz"
            
            seg_mask, mgmt_pred, feature_vector = model_inferer_test(image)
            prob = torch.sigmoid(seg_mask)
            seg = prob[0].detach().cpu().numpy()
            mgmt_pred = mgmt_pred.detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4

            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(seg_mask)

            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            print("Inference done on case {}, Prediction = {} ({}), Dice (ET, WT, TC): {}".format(img_name, mgmt_pred[0],  mgmt_target.detach().cpu().numpy()[0], run_acc.avg))

            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))
            dice_et += run_acc.avg[0]
            dice_wt += run_acc.avg[1]
            dice_tc += run_acc.avg[2]
            preds.append(mgmt_pred[0])
            labels.append(mgmt_target.detach().cpu().numpy())
            ids.append(num)

            np.save(os.path.join(output_directory, num+".npy"), feature_vector.detach().cpu().numpy()) #Feature Vector
            print(f"Feature Vector of {num}.npy saved")
        
        pred_df = pd.DataFrame.from_dict({
            "BraTS21ID" : ids,
            "MGMT_value" : preds,
            "Label" : labels
        })
        pred_df.to_csv(os.path.join(output_directory, "submission.csv"), index=False)

        print("\n\n---------------------------------------------------")
        print(f"ROC AUC Score on Test Dataset: {roc_auc_score(labels, preds)}")
        print("Dice Scores on Test Dataset")
        print(f"Dice ET: {dice_et/len(labels)}, Dice WT: {dice_wt/len(labels)}, Dice ET: {dice_tc/len(labels)}")
        print("---------------------------------------------------")


if __name__ == "__main__":
    main()
