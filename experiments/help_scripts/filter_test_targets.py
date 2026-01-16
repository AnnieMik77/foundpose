import json 

path = "/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/housecat/test_targets_bop19.json"

with open(path, 'r') as f:
    data = json.load(f)
print(f"Total number of test targets: {len(data)}")

metallic_ids = [5,15,25,35,45]
glass_ids = [6,16,26,36,46]

for dato in data:
    if dato['scene_id'] == 10:
        print(dato)

metallic_targets = [d for d in data if d['obj_id'] in metallic_ids]
glass_targets = [d for d in data if d['obj_id'] in glass_ids]

print(f"Number of metallic targets: {len(metallic_targets)}")
print(f"Number of glass targets: {len(glass_targets)}")

# store metallic targets
with open("/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/housecat/metallic_test_targets_bop19.json", 'w') as f:
    json.dump(metallic_targets, f)
# store glass targets
with open("/local2/homes/mikesann/multiview_proj/mock_data_dir/bop_datasets/housecat/glass_test_targets_bop19.json", 'w') as f:
    json.dump(glass_targets, f)