import os

import bop_toolkit_lib
import subprocess

folder_path = "/local2/homes/mikesann/multiview_proj/mock_data_dir/results/cosypose_baseline_eval/num_views_exp/tless_unknown_cams"

# recursively find all .csv files in the folder
csv_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

for csv_path in csv_files:
    print(f"Evaluating {csv_path}...")
    eval_dir = os.path.dirname(csv_path)
    print(f"Eval dir: {eval_dir}")
    # os.makedirs(eval_dir, exist_ok=True)

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
