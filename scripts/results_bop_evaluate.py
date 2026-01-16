import os

import bop_toolkit_lib
import subprocess
import bop_toolkit_lib.config as bop_config
from utils import misc

object_dataset = "ycbv"
inference_version = "featuremetric_refinement"
signature = misc.slugify(object_dataset) + "_{}".format(inference_version)


csv_path = os.path.join(
    bop_config.output_path, "inference", signature, f"foundpose_{object_dataset}-test.csv"
)
eval_dir = os.path.dirname(csv_path)

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
