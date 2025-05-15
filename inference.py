import os
from glob import glob
import math

import numpy as np
import torch
from torch.amp import autocast as autocast
from einops import rearrange
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from einops import repeat
import scipy
from scipy.ndimage import zoom

from model.maskformer import Maskformer
from model.knowledge_encoder import Knowledge_Encoder

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


EXPERIMENT_NAME = "mmSAT"
SLICE_START = 1400
SLICE_END = -1  # None means all slices

MODEL_CKPT = None

CT_MODEL_CKPT = (
    "/cachedata/yb107/results/logs/nano_CT_SAT_model/checkpoint/step_45000.pth"  # 45000
)
MR_MODEL_CKPT = "/cachedata/yb107/results/logs/nano_MRI_SAT_model/checkpoint/step_68000.pth"  # 68000
MS_MODEL_CKPT = "/cachedata/yb107/results/logs/nano_MS_SAT_model/checkpoint/step_177000.pth"  # 177000
US_MODEL_CKPT = (
    "/cachedata/yb107/results/logs/nano_US_SAT_model/checkpoint/step_76000.pth"  # 76000
)
PET_MODEL_CKPT = "/cachedata/yb107/results/logs/nano_PET_SAT_model/checkpoint/step_57000.pth"  # 57000

TEXT_ENCODER_CKPT = "/cachedata/yb107/models/text_encoder_cvpr25_v0.pth"

INPUT_DIR = "/scratch/railabs/ld258/dataset/public_dataset/CVPR_seg_2025/3D_val_npz/"
OUTPUT_DIR = f"/cachedata/yb107/inferences/{EXPERIMENT_NAME}"  # Avoid tailing slashes
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda", 0)


class EnsembleModel:
    def __init__(
        self,
        ct_ckpt: str,
        mri_ckpt: str,
        pet_ckpt: str,
        us_ckpt: str,
        ms_ckpt: str,
        model_class,
    ):
        # map modality name → checkpoint path
        self.ckpt_map = {
            "ct": ct_ckpt,
            "mri": mri_ckpt,
            "pet": pet_ckpt,
            "us": us_ckpt,
            "microscopy": ms_ckpt,
        }
        self.model = model_class
        self.current_modality = None

        for modality in self.ckpt_map.keys():
            checkpoint = torch.load(
                self.ckpt_map[modality], map_location=device, weights_only=False
            )
            # Remove 'module.' prefix from keys in checkpoint
            new_state_dict = {}
            for key, value in checkpoint["model_state_dict"].items():
                if "mid_mask_embed_proj" in key:
                    continue
                if key.startswith("module."):
                    new_state_dict[key[7:]] = value  # Remove first 7 chars ('module.')
                else:
                    new_state_dict[key] = value
            checkpoint["model_state_dict"] = new_state_dict

            # save the checkpoint
            torch.save(
                checkpoint,
                os.path.join(
                    OUTPUT_DIR,
                    f"checkpoint_{modality}.pt",
                ),
            )
            # update the ckpt_map
            self.ckpt_map[modality] = os.path.join(
                OUTPUT_DIR,
                f"checkpoint_{modality}.pt",
            )

    def __call__(
        self,
        modality: str,
    ):
        if modality != self.current_modality:
            self.current_modality = modality
            checkpoint = torch.load(
                self.ckpt_map[modality], map_location=device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        return self.model


def split_3d(image_tensor, crop_size=[288, 288, 96]):
    # C H W D
    interval_h, interval_w, interval_d = (
        crop_size[0] // 2,
        crop_size[1] // 2,
        crop_size[2] // 2,
    )
    split_idx = []
    split_patch = []

    c, h, w, d = image_tensor.shape
    h_crop = max(math.ceil(h / interval_h) - 1, 1)
    w_crop = max(math.ceil(w / interval_w) - 1, 1)
    d_crop = max(math.ceil(d / interval_d) - 1, 1)

    for i in range(h_crop):
        h_s = i * interval_h
        h_e = h_s + crop_size[0]
        if h_e > h:
            h_s = h - crop_size[0]
            h_e = h
            if h_s < 0:
                h_s = 0
        for j in range(w_crop):
            w_s = j * interval_w
            w_e = w_s + crop_size[1]
            if w_e > w:
                w_s = w - crop_size[1]
                w_e = w
                if w_s < 0:
                    w_s = 0
            for k in range(d_crop):
                d_s = k * interval_d
                d_e = d_s + crop_size[2]
                if d_e > d:
                    d_s = d - crop_size[2]
                    d_e = d
                    if d_s < 0:
                        d_s = 0
                split_idx.append([h_s, h_e, w_s, w_e, d_s, d_e])
                split_patch.append(image_tensor[:, h_s:h_e, w_s:w_e, d_s:d_e])

    return split_patch, split_idx


def pad_if_necessary(image):
    """
    Pad image to 288 288 96
    """
    c, h, w, d = image.shape
    croph, cropw, cropd = [288, 288, 96]
    pad_in_h = 0 if h >= croph else croph - h
    pad_in_w = 0 if w >= cropw else cropw - w
    pad_in_d = 0 if d >= cropd else cropd - d

    # Store padding information
    padding_info = (pad_in_h, pad_in_w, pad_in_d)

    if pad_in_h + pad_in_w + pad_in_d > 0:
        pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
        image = F.pad(image, pad, "constant", 0)  # chwd

    return image, padding_info


def remove_padding(padded_image, padding_info):
    """
    Removes padding
    """
    pad_in_h, pad_in_w, pad_in_d = padding_info

    if len(padded_image.shape) == 4:
        if isinstance(padded_image, torch.Tensor):
            return padded_image[
                :,
                : padded_image.shape[1] - pad_in_h,
                : padded_image.shape[2] - pad_in_w,
                : padded_image.shape[3] - pad_in_d,
            ]
        else:  # numpy array
            return padded_image[
                :,
                : padded_image.shape[1] - pad_in_h,
                : padded_image.shape[2] - pad_in_w,
                : padded_image.shape[3] - pad_in_d,
            ]
    else:
        if isinstance(padded_image, torch.Tensor):
            return padded_image[
                : padded_image.shape[0] - pad_in_h,
                : padded_image.shape[1] - pad_in_w,
                : padded_image.shape[2] - pad_in_d,
            ]
        else:  # numpy array
            return padded_image[
                : padded_image.shape[0] - pad_in_h,
                : padded_image.shape[1] - pad_in_w,
                : padded_image.shape[2] - pad_in_d,
            ]


def respace_image(
    image: np.ndarray, current_spacing: np.ndarray, target_spacing: np.ndarray
) -> np.ndarray:
    # Calculate zoom factors (ratio between current and target spacing)
    zoom_factors = np.array(current_spacing) / np.array(target_spacing)
    # Apply resampling using scipy.ndimage.zoom
    # order=1 uses linear interpolation
    resampled_image = scipy.ndimage.zoom(image, zoom_factors, order=1)
    return resampled_image


def get_data_dicts(dir, slice_start=None, slice_end=None):
    # Get all npz files in the directory
    npz_files = glob(os.path.join(dir, "*.npz"))
    npz_files = sorted(npz_files)

    if slice_start is not None and slice_end is not None:
        npz_files = npz_files[slice_start:slice_end]
        logging.info(
            f"Loading {len(npz_files)} npz files from {slice_start} to {slice_end}"
        )
    else:
        logging.info(f"Found {len(npz_files)} npz files in {dir}")

    data_dicts = []
    for npz_file in npz_files:
        data_dict = read_npz_data(npz_file)
        data_dicts.append(data_dict)
    return data_dicts


def read_npz_data(npz_file):

    # npz_file = glob(f"{INPUT_DIR}/*.npz")[0]
    data = np.load(npz_file, allow_pickle=True)

    raw_image = data["imgs"].astype(np.float32)  # 0~255
    raw_d, raw_h, raw_w = raw_image.shape
    # d h w -> h w d
    image = rearrange(raw_image, "d h w -> h w d")  # [h, w, d]

    # do respacing for CT
    if "CT_" in npz_file:
        image = respace_image(image, data["spacing"], target_spacing=[1.0, 1.0, 3.0])

    # padding
    image = repeat(image, "h w d -> c h w d", c=3)
    image = torch.tensor(image)
    image, padding_info = pad_if_necessary(image)  # [3, h, w, d]
    _, h, w, d = image.shape

    text_prompts = data["text_prompts"].item()
    del text_prompts["instance_label"]
    texts = list(text_prompts.values())  # ['xxx', ...]
    values = list(text_prompts.keys())  # [1, 2, ...]

    # Determine modality based on texts:
    terminologies = {
        "ct": ["CT", "CT_", "Computed Tomography", "CT Scan"],
        "mri": ["MRI", "MR", "Magnetic Resonance Imaging", "MRI Scan"],
        "us": ["US", "Ultrasound", "Ultrasound Scan"],
        "pet": ["PET", "Positron Emission Tomography", "PET Scan"],
        "microscopy": ["Microscopy", "Microscope", "Microscopy Scan"],
    }
    modality = None
    for key, vals in terminologies.items():
        if any(value in texts[0] for value in vals):
            modality = key
            break

    if modality is None:
        logging.warning(
            f"Modality not found in texts. Checking filename for modality: {npz_file}"
        )

        npz_file = os.path.basename(npz_file)
        if npz_file.startswith("CT"):
            modality = "ct"
        elif npz_file.startswith("MR"):
            modality = "mri"
        elif npz_file.startswith("US"):
            modality = "us"
        elif npz_file.startswith("PET"):
            modality = "pet"
        elif npz_file.startswith("Microscopy"):
            modality = "microscopy"
        else:
            raise ValueError(f"Unknown modality for file {npz_file}")

    patches, y1y2_x1x2_z1z2_ls = split_3d(
        image, crop_size=[288, 288, 96]
    )  # [[3, 288, 288, 96], ...]  # [[y1, y2, x1, x2, z1, z2], ...]

    return {
        "npz_file_name": os.path.basename(npz_file),
        "modality": modality,
        "texts": texts,
        "values": values,
        "original_shape": (raw_h, raw_w, raw_d),
        "current_shape": (h, w, d),
        "patches": patches,
        "y1y2_x1x2_z1z2_ls": y1y2_x1x2_z1z2_ls,
        "padding_info": padding_info,
        # "raw_image": raw_image,
    }


def compute_gaussian(
    tile_size,
    sigma_scale: float = 1.0 / 8,
    value_scaling_factor: float = 10,
    dtype=np.float16,
):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)

    # gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map = (
        gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    )
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0]
    )

    return gaussian_importance_map


def main():

    # set gpu
    device = torch.device("cuda", 0)

    # load model
    model_class = Maskformer("UNET", [288, 288, 96], [32, 32, 32], False)
    model_class = model_class.to(device)

    if MODEL_CKPT is not None:
        checkpoint = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
        model_class.load_state_dict(checkpoint["model_state_dict"])
        model = model_class
        model.eval()

    else:
        ensemble_router = EnsembleModel(
            ct_ckpt=CT_MODEL_CKPT,
            mri_ckpt=MR_MODEL_CKPT,
            pet_ckpt=PET_MODEL_CKPT,
            us_ckpt=US_MODEL_CKPT,
            ms_ckpt=MS_MODEL_CKPT,
            model_class=model_class,
        )

    ## MAke sure to load the model with the same architecture as the checkpoint
    ## Go to May13th notebook to fix the checkpoint
    # Remove 'module.' prefix from keys in checkpoint
    # new_state_dict = {}
    # for key, value in checkpoint["model_state_dict"].items():
    #     if "mid_mask_embed_proj" in key:
    #         continue
    #     if key.startswith("module."):
    #         new_state_dict[key[7:]] = value  # Remove first 7 chars ('module.')
    #     else:
    #         new_state_dict[key] = value
    # checkpoint["model_state_dict"] = new_state_dict
    # model.load_state_dict(checkpoint["model_state_dict"])

    # load text encoder
    text_encoder = Knowledge_Encoder()
    text_encoder = text_encoder.to(device)
    checkpoint = torch.load(TEXT_ENCODER_CKPT, map_location=device)
    # Remove 'module.' prefix from keys in checkpoint
    new_state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value  # Remove first 7 chars ('module.')
        else:
            new_state_dict[key] = value
    checkpoint["model_state_dict"] = new_state_dict
    text_encoder.load_state_dict(checkpoint["model_state_dict"], strict=False)
    text_encoder.eval()

    with torch.no_grad():

        # gaussian kernel to accumulate predcition
        gaussian = torch.tensor(compute_gaussian((288, 288, 96))).to(device)  # hwd

        # load data
        logging.info(f"Loading data from {INPUT_DIR}")
        data_dicts = get_data_dicts(INPUT_DIR, SLICE_START, SLICE_END)

        logging.info(f"Loaded {len(data_dicts)} data files")
        c = 0
        for data_dict in data_dicts:
            # load and process inference data
            # data_dict = read_npz_data()
            logging.info(f"Infering {c} / {len(data_dicts)}")
            c += 1
            # Extract individual values from dictionary
            file_name = data_dict["npz_file_name"]
            modality = data_dict["modality"]

            text_prompts = data_dict["texts"]
            label_values = data_dict["values"]

            original_shape = data_dict["original_shape"]
            current_shape = data_dict["current_shape"]
            batched_patches = data_dict["patches"]
            batched_y1y2_x1x2_z1z2 = data_dict["y1y2_x1x2_z1z2_ls"]
            padding_info = data_dict["padding_info"]
            # raw_image = data_dict["raw_image"]

            modality_code_dict = {"ct": 0, "mri": 1, "us": 2, "pet": 3, "microscopy": 4}
            modality_code = torch.tensor([modality_code_dict[modality]]).to(device)

            if MODEL_CKPT is not None:
                model = model
            else:
                model = ensemble_router(modality)

            h, w, d = current_shape
            n = len(text_prompts)
            prediction = torch.zeros((n, h, w, d))
            accumulation = torch.zeros((n, h, w, d))
            with autocast(device_type=device.type):

                # encode text prompts
                queries = text_encoder(
                    text_prompts, modality_code
                )  # convert text prompts to embeds
                torch.cuda.empty_cache()

                # for each batch of patches, query with all labels
                for patches, y1y2_x1x2_z1z2_ls in zip(
                    batched_patches, batched_y1y2_x1x2_z1z2
                ):  # [c, h, w, d]
                    patches = patches.unsqueeze(0).to(device=device)  # [b, c, h, w, d]
                    prediction_patch = model(
                        queries=queries, image_input=patches, train_mode=False
                    )
                    prediction_patch = torch.sigmoid(prediction_patch)  # bnhwd
                    prediction_patch = prediction_patch.detach()  # .cpu().numpy()

                    # fill in
                    y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls
                    # gaussian accumulation
                    tmp = (
                        prediction_patch[0, :, : y2 - y1, : x2 - x1, : z2 - z1]
                        * gaussian[: y2 - y1, : x2 - x1, : z2 - z1]
                    )  # on gpu
                    prediction[:, y1:y2, x1:x2, z1:z2] += tmp.cpu()
                    accumulation[:, y1:y2, x1:x2, z1:z2] += gaussian[
                        : y2 - y1, : x2 - x1, : z2 - z1
                    ].cpu()

                # avg
                prediction = prediction / accumulation
                prediction = torch.where(prediction > 0.5, 1.0, 0.0)
                prediction = prediction.numpy()

            # save prediction
            results = np.zeros((h, w, d))  # hwd
            for j, (text, value) in enumerate(zip(text_prompts, label_values)):
                results += prediction[j, :, :, :] * int(value)
            results = remove_padding(results, padding_info)
            # Check if the current shape is different from original shape (respaced) and resize if needed
            current_h, current_w, current_d = results.shape
            original_h, original_w, original_d = original_shape
            if (
                current_h != original_h
                or current_w != original_w
                or current_d != original_d
            ):
                # Use scipy's resize function to restore to original shape
                zoom_factors = (
                    original_h / current_h,
                    original_w / current_w,
                    original_d / current_d,
                )
                # Use nearest neighbor interpolation (order=0) to preserve label values
                results = zoom(results, zoom_factors, order=0)
                logging.info(
                    f"Resized segmentation from {(current_h, current_w, current_d)} to {(original_h, original_w, original_d)}"
                )
            results = rearrange(results, "h w d -> d h w")
            np.savez_compressed(f"{OUTPUT_DIR}/{file_name}", segs=results)
            logging.info(f"Saved segmentation to {os.path.join(OUTPUT_DIR, file_name)}")

    # #
    # # Save as NIfTI (nii.gz) file with labeled regions
    # #

    # # Create NIfTI file from results
    # import nibabel as nib
    # nifti_img = nib.Nifti1Image(results, np.eye(4))

    # # Save the NIfTI file
    # nii_output_path = os.path.join("./outputs", os.path.splitext(file_name)[0] + "_pred.nii.gz")
    # nib.save(nifti_img, nii_output_path)

    # # Create a text fil  e with label descriptions
    # label_description_path = os.path.join("./outputs", os.path.splitext(file_name)[0] + "_labels.txt")
    # with open(label_description_path, "w") as f:
    #     f.write("Label descriptions:\n")
    #     for value, text in zip(label_values, text_prompts):
    #         f.write(f"Label {value}: {text}\n")

    # print(f"Saved NIfTI file to {nii_output_path}")
    # print(f"Saved label descriptions to {label_description_path}")

    # #
    # # Save raw image as NIfTI (nii.gz) file
    # #

    # # Create NIfTI file from raw image
    # raw_nifti_img = nib.Nifti1Image(raw_image, np.eye(4))

    # # Save the raw image NIfTI file
    # raw_nii_output_path = os.path.join("./outputs", os.path.splitext(file_name)[0] + "_img.nii.gz")
    # nib.save(raw_nifti_img, raw_nii_output_path)

    # print(f"Saved raw image NIfTI file to {raw_nii_output_path}")

    # #
    # # Save raw image slices as PNG files
    # #

    # # Create directory for PNG slices
    # png_output_dir = os.path.join("./outputs", os.path.splitext(file_name)[0] + "_slices")
    # os.makedirs(png_output_dir, exist_ok=True)

    # # Save each slice as PNG
    # import matplotlib.pyplot as plt
    # for slice_idx in range(raw_image.shape[0]):
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(raw_image[slice_idx], cmap='gray')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(png_output_dir, f"slice_{slice_idx:03d}.png"),
    #                 bbox_inches='tight', pad_inches=0, dpi=150)
    #     plt.close()

    # print(f"Saved {raw_image.shape[0]} PNG slices to {png_output_dir}")

    # #
    # # Test Evaluation DSC
    # #

    # # Load ground truth and segmentation masks
    # gt_path = os.path.join("./gts", file_name)
    # seg_path = os.path.join("./outputs", file_name)
    # gt_npz = np.load(gt_path, allow_pickle=True)['gts']
    # seg_npz = np.load(seg_path, allow_pickle=True)['segs']

    # # Calculate DSC
    # from evaluate.SurfaceDice import compute_dice_coefficient
    # def compute_multi_class_dsc(gt, seg):
    #     dsc = []
    #     for i in np.unique(gt)[1:]: # skip bg
    #         gt_i = gt == i
    #         seg_i = seg == i
    #         dsc.append(compute_dice_coefficient(gt_i, seg_i))
    #         print("dsc", dsc[-1])
    #     return np.mean(dsc)
    # dsc = compute_multi_class_dsc(gt_npz, seg_npz)
    # print('all dice:', dsc)


if __name__ == "__main__":
    main()
