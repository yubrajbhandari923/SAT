import os
import json
import pickle
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sampling_script.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ========== CONFIGURATION ==========
BASE_DIR = Path(
    "/scratch/railabs/ld258/dataset/public_dataset/CVPR_seg_2025/3D_train_npz_all/"
)
METADATA_JSON = Path(
    "/home/yb107/cvit-work/cvpr2025/SAT/data/challenge_data/CVPR25_TextSegFMData_with_class.json"
)
TMP_DIR = Path("/home/yb107/cvit-work/cvpr2025/SAT/notebook/tmp")
DS_CONFIG_JSON = Path(
    "/home/yb107/cvit-work/cvpr2025/SAT/data/dataset_config/cvpr25.json"
)

TARGET_FRACTION = 0.082  # 10%
MIN_PER_DATASET = 5
MIN_PER_MODALITY = 50
RANDOM_SEED = 42
N_WORKERS = 55

logging.info(f"Using {N_WORKERS} workers for parallel processing")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ensure tmp directory exists
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ========== STEP 1: LOAD METADATA ==========
with METADATA_JSON.open() as f:
    raw_meta = json.load(f)

with DS_CONFIG_JSON.open() as f:
    ds_config = json.load(f)

metadata = {}
for ds_name, info in raw_meta.items():
    label_ids = sorted(int(k) for k in info.keys() if k.isdigit())
    instance_flag = bool(info.get("instance_label", 0))
    metadata[ds_name] = {"label_ids": label_ids, "instance_label": instance_flag}
logging.info("Step 1 Done: metadata loaded")


# ========== STEP 2: DISCOVER ALL CANDIDATE FILES (with checkpoint) ==========
files_by_modality_path = TMP_DIR / "files_by_modality.pkl"
if files_by_modality_path.exists():
    with files_by_modality_path.open("rb") as f:
        files_by_modality = pickle.load(f)
    logging.info("Loaded Step 2 checkpoint: files_by_modality")
else:
    files_by_modality = defaultdict(lambda: defaultdict(list))
    for modality_dir in BASE_DIR.iterdir():
        if not modality_dir.is_dir():
            continue
        modality = modality_dir.name
        for ds_dir in modality_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            ds_name = ds_dir.name
            if ds_name not in metadata:
                logging.info(f"Warning: {ds_name} not in metadata, skipping")
                continue
            if ds_name not in ds_config:
                logging.info(f"Warning: {ds_name} not in ds_config, skipping")
                continue
            for npz_path in ds_dir.glob("*.npz"):
                files_by_modality[modality][ds_name].append(npz_path)
    with files_by_modality_path.open("wb") as f:
        files_by_modality = {k: dict(v) for k, v in files_by_modality.items()}
        pickle.dump(files_by_modality, f)
    logging.info("Step 2 Done and checkpoint saved")


# ========== STEP 3: PRECOMPUTE WEIGHTS AND LABEL EXISTENCE (with checkpoint & corrupted skip) ==========
file_info_path = TMP_DIR / "file_info.pkl"
if file_info_path.exists():
    with file_info_path.open("rb") as f:
        file_info = pickle.load(f)
    total_candidates = len(file_info)
    target_n = int(total_candidates * TARGET_FRACTION)
    logging.info(
        f"Loaded Step 3 checkpoint: total candidates={total_candidates}, target={target_n}"
    )
else:
    # --- helper for multiprocessing ---------------------------------
    def _process_file(args):
        """Return (ok, modality, ds, path, exist_list, weight)."""
        modality, ds, path, label_ids, inst = args
        try:
            arr = np.load(path)
            gt = arr["gts"]
            if inst:
                gt = (gt > 0).astype(np.uint8)
            exist = [gt.any() if inst else bool((gt == lid).any()) for lid in label_ids]
            weight = 1 + sum(exist)
            return True, modality, ds, path, exist, weight
        except Exception as e:
            return False, modality, ds, path, None, None

    # ----------------------------------------------------------------

    jobs = []
    for modality, ds_map in files_by_modality.items():
        for ds_name, paths in ds_map.items():
            label_ids = metadata[ds_name]["label_ids"]
            instance_flag = metadata[ds_name]["instance_label"]
            for p in paths:
                jobs.append((modality, ds_name, p, label_ids, instance_flag))

    file_info = {}
    corrupted_cnt = 0
    valid_paths = defaultdict(lambda: defaultdict(list))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for ok, modality, ds_name, p, exist, weight in pool.map(_process_file, jobs):
            if ok:
                file_info[(modality, ds_name, p)] = {"exist": exist, "weight": weight}
                valid_paths[modality][ds_name].append(p)
            else:
                corrupted_cnt += 1

    # overwrite files_by_modality with cleaned lists
    files_by_modality = valid_paths
    # save both cleaned structures
    # Convert defaultdict to regular dict for serialization
    files_by_modality = {k: dict(v) for k, v in files_by_modality.items()}

    pickle.dump(file_info, file_info_path.open("wb"))
    pickle.dump(files_by_modality, files_by_modality_path.open("wb"))

    total_candidates = len(file_info)
    target_n = int(total_candidates * TARGET_FRACTION)
    logging.info(
        f"Step 3 done: {total_candidates} usable files, {corrupted_cnt} corrupted removed (parallel x{N_WORKERS})"
    )


# ========== STEP 4 & 5: SAMPLING (with checkpoint) ==========
selected_path = TMP_DIR / "selected.pkl"
if selected_path.exists():
    with selected_path.open("rb") as f:
        selected = pickle.load(f)
    logging.info(f"Loaded sampling checkpoint: {len(selected)} entries selected")
else:
    selected = set()

    # Phase 1a: Ensure MIN_PER_DATASET per dataset
    for modality, ds_map in files_by_modality.items():
        for ds_name, paths in ds_map.items():
            candidates = [
                key for key in file_info if key[0] == modality and key[1] == ds_name
            ]
            if not candidates:
                continue
            k = min(MIN_PER_DATASET, len(candidates))
            weights = np.array(
                [file_info[key]["weight"] for key in candidates], dtype=float
            )
            probs = weights / weights.sum()
            chosen = np.random.choice(len(candidates), size=k, replace=False, p=probs)
            for idx in chosen:
                selected.add(candidates[idx])

    # Phase 1b: Ensure MIN_PER_MODALITY per modality
    for modality, ds_map in files_by_modality.items():
        already = [key for key in selected if key[0] == modality]
        need = max(0, MIN_PER_MODALITY - len(already))
        if need == 0:
            continue
        remaining = [
            key for key in file_info if key[0] == modality and key not in selected
        ]
        weights = np.array([file_info[key]["weight"] for key in remaining], dtype=float)
        probs = weights / weights.sum()
        chosen = np.random.choice(
            len(remaining), size=min(need, len(remaining)), replace=False, p=probs
        )
        for idx in chosen:
            selected.add(remaining[idx])

    # Phase 1c: Prune if overshoot
    if len(selected) > target_n:
        to_remove = len(selected) - target_n
        ds_sizes = {
            ds: len(files_by_modality[mod][ds])
            for mod, ds_map in files_by_modality.items()
            for ds in ds_map
        }
        selected_list = list(selected)
        rem_weights = np.array([ds_sizes[key[1]] for key in selected_list], dtype=float)
        rem_probs = rem_weights / rem_weights.sum()
        remove_idxs = np.random.choice(
            len(selected_list), size=to_remove, replace=False, p=rem_probs
        )
        for idx in remove_idxs:
            selected.remove(selected_list[idx])

    # Phase 2: Fill to target
    remaining_slots = target_n - len(selected)
    if remaining_slots > 0:
        pool = [key for key in file_info if key not in selected]
        weights = np.array([file_info[key]["weight"] for key in pool], dtype=float)
        probs = weights / weights.sum()
        chosen = np.random.choice(
            len(pool), size=min(remaining_slots, len(pool)), replace=False, p=probs
        )
        for idx in chosen:
            selected.add(pool[idx])

    with selected_path.open("wb") as f:
        pickle.dump(selected, f)
    logging.info(f"Sampling done: {len(selected)} entries selected. Checkpoint saved")


# ========== STEP 6: WRITE OUTPUT JSONL ==========
out_path = Path(
    "/home/yb107/cvit-work/cvpr2025/SAT/data/challenge_data/custom_10percent_sample.jsonl"
)
with out_path.open("w") as out_f:
    for modality, ds_name, p in sorted(selected, key=lambda x: (x[0], x[1], str(x[2]))):
        info = file_info[(modality, ds_name, p)]
        record = {
            "data": str(p),
            "dataset": ds_name,
            "modality": modality,
            "label_existance": [bool(v) for v in info["exist"]],
        }
        out_f.write(json.dumps(record) + "\n")

logging.info(f"Wrote {len(selected)} entries to {out_path.resolve()}")
