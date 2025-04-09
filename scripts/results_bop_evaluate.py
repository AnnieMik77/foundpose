import os

import bop_toolkit_lib
import subprocess

csv_path = "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/inference/ycbv_v_foundpose_vitl_layer18/coarse_ycbv-test.csv"
eval_dir = "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/inference/ycbv_v_foundpose_vitl_layer18"
os.makedirs(eval_dir, exist_ok=True)

bop_path = os.path.dirname(bop_toolkit_lib.__file__).split("/bop_toolkit_lib")[0]
script_path = os.path.join(bop_path, "scripts", "eval_bop19_pose.py")
command = [
    "python", 
    script_path, 
    "--renderer_type=vispy", 
    f"--result_filenames={csv_path}",
    f"--results_path={eval_dir}",
    f"--eval_path={eval_dir}",
    f"--targets_filename=test_targets_bop19.json",
    f"--num_workers=1"
]

subprocess.run(command)
