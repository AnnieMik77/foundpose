#!/usr/bin/env python3
import pandas as pd

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import copy

from utils import (
    projector_util,
    repre_util,
    vis_base_util,
    renderer_base,
    render_vis_util,
    logging, 
    structs, 
    misc, 
    geometry
)

from utils.misc import tensor_to_array, array_to_tensor

logger: logging.Logger = logging.get_logger()
import os
import json

# something for parsing and reading csv
import csv


path_bop_test_targets = "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/tless/test_targets_bop19.json"
path_bop_eval_outputs = "/local2/homes/mikesann/multiview_proj/mock_data_dir/results/tless_multi_for_downloaded"
path_bop_dataset = "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/tless/test_primesense"
path_inference = "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/inference"

test_targets = json.load(open(path_bop_test_targets))
def get_unique_scene_ids(test_targets):
    scene_ids = []
    for target in test_targets:
        scene_id = target["scene_id"]
        if scene_id not in scene_ids:
            scene_ids.append(scene_id)
    return scene_ids

def get_images_per_scene(scene_id, test_targets):
    images = []
    for target in test_targets:
        if target["scene_id"] == scene_id:
            images.append(target["im_id"])
    return images



def get_prediction_error(prediction_csv_path, setting="mspd"):
    # load the predictions csv file
    experiment_csv = os.path.basename(prediction_csv_path)

    # Load the CSV file
    df = pd.read_csv(prediction_csv_path)

    # Convert to JSON-like structure (list of dictionaries)
    data = df.to_dict(orient="records")

    path_experiment = os.path.join(path_bop_eval_outputs, experiment_csv.replace(".csv", ""))
    mspd_error_path = os.path.join(path_experiment, f"error={setting}_ntop=-1")


    # get the scene ids
    scene_ids = get_unique_scene_ids(test_targets)
    errors_all = []
    for scene_id in scene_ids:
        mspd_per_scene = os.path.join(mspd_error_path, f"errors_{scene_id:06d}.json")
        mspd_errors = json.load(open(mspd_per_scene))

        for i in range(len(mspd_errors)):
            mspd_error_obj = mspd_errors[i]
            im_id = mspd_error_obj["im_id"]
            obj_id = mspd_error_obj["obj_id"]
            mspd_error = list(mspd_error_obj["errors"].values())[0][0]
            error_object = {
                "scene_id": scene_id,
                "im_id": im_id,
                "obj_id": obj_id,
                "error": mspd_error
            }
            errors_all.append(error_object)

    return errors_all



exp1_csv = "/local2/homes/mikesann/multiview_proj/mock_data_dir/results/tless_multi_for_downloaded/foundpose_multiview_nviews=4_928040/tless.bop19/bop_evaluation/foundpose-multiview-nviews=4-928040-ba-input_tless-test.csv"
exp2_csv = "/local2/homes/mikesann/multiview_proj/mock_data_dir/results/tless_multi_for_downloaded/foundpose_mw/foundpose-cosy-foundpose-multiview-nviews=4-928040_tless-test.csv"
metric = "mssd"

exp_1_errors = get_prediction_error(exp1_csv, setting=metric)
if exp2_csv is not None:
    exp_2_errors = get_prediction_error(exp2_csv, setting=metric)

# Compare the errors from the two experiments
# when there is the largest decrease in error, print the scene_id, im_id, obj_id, and error

# create folder for visualizations of the worst predictions
os.makedirs("worst_predictions", exist_ok=True)
os.makedirs(f"worst_predictions/{exp1_csv.split('/')[-1].replace('.csv', '')}", exist_ok=True)
os.makedirs(f"worst_predictions/{exp2_csv.split('/')[-1].replace('.csv', '')}", exist_ok=True)

if exp2_csv is not None:
    for i in range(len(exp_1_errors)):
        error_1 = exp_1_errors[i]
        error_2 = exp_2_errors[i]
        assert error_1["scene_id"] == error_2["scene_id"]
        assert error_1["im_id"] == error_2["im_id"]
        assert error_1["obj_id"] == error_2["obj_id"]
        error_diff = error_1["error"] - error_2["error"]
        error_1["error_diff"] = error_diff
        error_1["error_2"] = error_2["error"]

    exp_1_errors.sort(key=lambda x: x["error_diff"], reverse=True)
    worst_prediction_ids = exp_1_errors[:15]


else:
    exp_1_errors.sort(key=lambda x: x["error"], reverse=True)
    worst_prediction_ids = exp_1_errors[:50]
    from pprint import pprint
    pprint(worst_prediction_ids)





from pprint import pprint
pprint(worst_prediction_ids)

# Show the worst images
images_path = '~/multiview_proj/mock_data_dir/bop_datasets/inference/tless_v_masked_bow_gtdet_nodrift'
for pred in worst_prediction_ids:
    scene_id = pred["scene_id"]
    im_id = pred["im_id"]
    obj_id = pred["obj_id"]
    image_path = os.path.join(images_path, str(obj_id), f"{scene_id}_{im_id}_{obj_id}_0_0.png")
    print(image_path)