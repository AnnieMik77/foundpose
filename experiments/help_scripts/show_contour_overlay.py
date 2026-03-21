import os
import numpy as np
import pandas as pd
import cv2
import argparse
from bop_toolkit_lib import inout, dataset_params
import bop_toolkit_lib.config as bop_config
import bop_toolkit_lib.misc as bop_misc

from utils.renderer_builder import build_renderer
from utils.data_util import load_image  # adjust import if needed

def get_contour(mask, color=(0, 255, 0), dilate_iterations=1):
    mask_uint8 = (mask.astype(np.uint8) * 255)
    canny = cv2.Canny(mask_uint8, threshold1=30, threshold2=100)
    kernel = np.ones((3, 3), np.uint8)
    canny = cv2.dilate(canny, kernel, iterations=dilate_iterations)
    return canny

def overlay_contour(img, contour, color=(0, 255, 0)):
    img_contour = img.copy()
    img_contour[contour > 0] = color
    return img_contour

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, default="ycbv")
    parser.add_argument('--scene_id', type=int, required=True, default=53)
    parser.add_argument('--im_id', type=int, required=True, default=1)
    parser.add_argument('--csv_path', type=str, required=True, default="/local2/homes/mikesann/multiview_proj/mock_data_dir/results/ycbv_downloaded/final_eval_render/cnos-fastsammegapose-multihyp-10_ycbv-test_7c18b17c-c7e8-442b-ba6d-3f2235eec6e3.csv")
    args = parser.parse_args()

    # Load image
    datasets_path = bop_config.datasets_path
    bop_test_split_props = dataset_params.get_split_params(
        datasets_path=datasets_path,
        dataset_name=args.dataset,
        split="val"
    )
    img_path = os.path.join(bop_test_split_props['dataset_path'], f"{args.scene_id:06d}", f"{args.im_id:06d}.png")
    img = inout.load_im(img_path)  # shape: H x W x 3


    # Load BOP dataset info
    bop_dataset_obj = BOPDataset(opts, logger=logger)
    bop_dataset_obj.load_split("test")


    # Load predictions
    df = pd.read_csv(args.csv_path)
    preds = df[(df['scene_id'] == args.scene_id) & (df['im_id'] == args.im_id)]

    # Build renderer
    renderer = build_renderer()  # adjust params as needed

    # Overlay contours for each prediction
    for _, pred in preds.iterrows():
        obj_id = int(pred['obj_id'])
        R = np.array(pred['R'].split(), dtype=np.float32).reshape(3, 3)
        t = np.array(pred['t'].split(), dtype=np.float32)
        mask = renderer.render_object_mask(obj_id=obj_id, R=R, t=t)
        contour = get_contour(mask)
        img = overlay_contour(img, contour)

    # Show result
    cv2.imshow("Contours Overlay", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()