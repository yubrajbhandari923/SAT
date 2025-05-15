import os

join = os.path.join
import shutil
import time
import torch
import argparse
from collections import OrderedDict
import pandas as pd
import numpy as np
from skimage import segmentation
from scipy.optimize import linear_sum_assignment
import cc3d
import SimpleITK as sitk

from glob import glob

from SurfaceDice import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
    compute_dice_coefficient,
)


def compute_multi_class_dsc(gt, seg, label_ids):
    dsc = [None] * len(label_ids)
    for idx, i in enumerate(label_ids):
        gt_i = gt == i
        seg_i = seg == i
        dsc[idx] = compute_dice_coefficient(gt_i, seg_i)

    return np.mean(dsc)


def compute_multi_class_nsd(gt, seg, spacing, label_ids, tolerance=2.0):
    nsd = []
    nsd = [None] * len(label_ids)
    for idx, i in enumerate(label_ids):
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(gt_i, seg_i, spacing_mm=spacing)
        nsd[idx] = compute_surface_dice_at_tolerance(surface_distance, tolerance)
    return np.mean(nsd)


def _label_overlap(x, y):
    """fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    x = x.ravel()
    y = y.ravel()

    # preallocate a 'contact map' matrix
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _intersection_over_union(masks_true, masks_pred):
    """intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou, th):
    """true positive at threshold th

    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp


def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)
    if num_inst_seg > 0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
        # for k,th in enumerate(threshold):
        tp = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        # logging.info('No segmentation results!')
        tp = 0
        fp = 0
        fn = 0

    return tp, fp, fn


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


EXPERIMENT_NAME = "mmSAT"

INPUT_DIR = "/scratch/railabs/ld258/dataset/public_dataset/CVPR_seg_2025/3D_val_npz/"
OUTPUT_DIR = f"/cachedata/yb107/inferences/{EXPERIMENT_NAME}"  # Avoid tailing slashes

GT_PATH = "/scratch/railabs/ld258/dataset/public_dataset/CVPR_seg_2025/3D_val_gt/3D_val_gt_text"  # validation ground truth directory

test_cases = sorted(
    glob(join(INPUT_DIR, "*.npz"))
)  # test cases are the input files in the input directory


# initialize the metric dictionary
metric = OrderedDict()
metric["CaseName"] = []
metric["RunningTime"] = []
metric["DSC"] = []
metric["NSD"] = []
metric["F1"] = []

# To obtain the running time for each case, testing cases are inferred one-by-one
for case in test_cases:
    real_running_time = 0

    # Metric calculation (DSC and NSD)
    seg_name = os.path.basename(case)
    gt_path = join(GT_PATH, seg_name)
    seg_path = join(OUTPUT_DIR, seg_name)

    try:
        # Load ground truth and segmentation masks
        gt_npz = np.load(gt_path, allow_pickle=True)["gts"]
        seg_npz = np.load(seg_path, allow_pickle=True)["segs"]

        gt_npz = gt_npz.astype(np.uint8)
        seg_npz = seg_npz.astype(np.uint8)

        # Calculate DSC and NSD
        img_npz = np.load(join(INPUT_DIR, case), allow_pickle=True)
        spacing = img_npz["spacing"]
        instance_label = img_npz["text_prompts"].item()["instance_label"]

        class_ids = sorted(
            [int(k) for k in img_npz["text_prompts"].item() if k != "instance_label"]
        )
        class_ids_array = np.array(class_ids, dtype=np.int32)

        if instance_label == 0:  # semantic masks
            # note: the semantic labels may not be sequential
            dsc = compute_multi_class_dsc(gt_npz, seg_npz, class_ids_array)
            nsd = compute_multi_class_nsd(gt_npz, seg_npz, spacing, class_ids_array)
            f1_score = np.nan
        elif instance_label == 1:  # instance masks
            # Calculate F1 instead
            if len(np.unique(seg_npz)) == 2:
                logging.info("converting segmentation to instance masks")
                # convert prediction masks from binary to instance
                tumor_inst, tumor_n = cc3d.connected_components(
                    seg_npz, connectivity=6, return_N=True
                )

                # put the tumor instances back to gt_data_ori
                seg_npz[tumor_inst > 0] = tumor_inst[tumor_inst > 0] + np.max(seg_npz)

            gt_npz = segmentation.relabel_sequential(gt_npz)[0]
            seg_npz = segmentation.relabel_sequential(seg_npz)[0]

            tp, fp, fn = eval_tp_fp_fn(
                gt_npz, seg_npz
            )  # default f1 overlap threshold is 0.5
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Set DSC and NSD to None for instance masks
            dsc = None
            nsd = None

        # save metrics
        metric["CaseName"].append(case)
        metric["RunningTime"].append(real_running_time)
        metric["DSC"].append(round(dsc, 4) if dsc is not None else np.nan)
        metric["NSD"].append(round(nsd, 4) if nsd is not None else np.nan)
        metric["F1"].append(round(f1_score, 4) if f1_score is not None else np.nan)

        logging.info(
            f"{case}: DSC={dsc if dsc is not None else np.nan}, NSD={nsd if nsd is not None else np.nan}, F1={f1_score}"
        )

    except Exception as e:
        logging.info(f"Error processing {case}: {e}")

for key in metric.keys():
    logging.info(f"{key}: length {len(metric[key])}: {metric[key]}")

# save the metrics to a CSV file
metric_df = pd.DataFrame(metric)

metric_df.to_csv(join(OUTPUT_DIR, "metrics.csv"), index=False)
logging.info(f"Metrics saved to {join(OUTPUT_DIR, 'metrics.csv')}")

# clean up
torch.cuda.empty_cache()
