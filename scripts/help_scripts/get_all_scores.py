import os

import bop_toolkit_lib
import subprocess

folder_path = "/local2/homes/mikesann/multiview_proj/mock_data_dir/results/tless_downloaded/num_view_exp/"

# recursively find all .csv files in the folder
result_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file == "scores_bop19.json":
            result_files.append(os.path.join(root, file))

import json
from pprint import pprint
results = {}
for score_file in result_files:
    # get the directory of the file
    # print(score_file)
    dir_path = os.path.dirname(score_file)

    # load the json file
    with open(score_file, "r") as f:
        data = json.load(f)

    results[dir_path] = data

pprint(results)
# # results to csv
# import pandas as pd
# df = pd.DataFrame.from_dict(results, orient='index')
# df.to_csv("results_ycbv_known_cameras.csv")



# store the results in a json file
# with open("results_tless_happypose_unknown_cameras.json", "w") as f:
#     json.dump(results, f, indent=4)