paths = [
    "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/inference/housecat_experiments/foundpose/glass",
    "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/inference/housecat_experiments/foundpose/metallic",
    "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/inference/housecat_experiments/cosypose/glass",
    "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/inference/housecat_experiments/cosypose/metallic",
    "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/inference/housecat_experiments/multiview/glass",
    "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/inference/housecat_experiments/multiview/metallic",
]

import json
import os
from pprint import pprint
results = {}
for eval_dir in paths:
    # recursively find all scores_bop19.json files in the folder
    result_files = []
    for root, dirs, files in os.walk(eval_dir):
        for file in files:
            if file == "scores_bop19.json" or file == "scores_bop24.json":
                result_files.append(os.path.join(root, file))

    for score_file in result_files:
        # get the directory of the file
        # print(score_file)
        dir_path = os.path.dirname(score_file)

        # load the json file
        with open(score_file, "r") as f:
            data = json.load(f)
        if "/multiview/" in dir_path:
            key_name = "multiview"
        elif "/cosypose/" in dir_path:
            key_name = "cosypose"
        else:
            key_name = "foundpose"

        if "glass" in eval_dir:
            key_name = f"{key_name}_glass"
        else:
            key_name = f"{key_name}_metallic"

        already = results.get(key_name, {})
        results[key_name] = {**already, **data}


      


# Convert to DataFrame
rows = []
for method, metrics in results.items():
    row = {'method': method}
    for k, v in metrics.items():
        # Standardize metric names (remove bop19/bop24 prefix)
        if "time_" in k:
            continue  # skip time metrics
        clean_name = k.split('_', 1)[-1]
        clean_name = clean_name.replace("average_recall", "AR")
        clean_name = clean_name.replace("mAP", "AP")
        # clean_name = k#.split('_', 1)[-1]
        v= 100 * v  # convert to percentage
        # round to 2 decimal places
        v = round(v, 1)
        row[clean_name] = v
    rows.append(row)

# store as CSV
import pandas as pd
df = pd.DataFrame(rows)

# sort by second element in method name
df = df.sort_values(by='method', key=lambda x: x.str.split('_').str[1])
csv_path = "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/inference/housecat_experiments/summary_glass_metallic.csv"
df.to_csv(csv_path, index=False)
pprint(df)