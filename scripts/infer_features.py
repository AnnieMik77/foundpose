#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gc
import time

from typing import List, NamedTuple, Optional, Tuple

import cv2

import numpy as np

import torch
import json
from utils.misc import array_to_tensor, tensor_to_array, tensors_to_arrays

from bop_toolkit_lib import inout, dataset_params
import bop_toolkit_lib.config as bop_config
import bop_toolkit_lib.misc as bop_misc


from utils import (
    corresp_util,
    config_util,
    eval_errors,
    eval_util,
    feature_util,
    infer_pose_util,
    knn_util,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util,
    vis_util,
    data_util,
    renderer_builder,
    json_util, 
    logging,
    misc,
    structs,
    template_util
)

from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import warp_depth_image, warp_image

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

logger: logging.Logger = logging.get_logger()


class InferOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    repre_version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None
    max_sym_disc_step: float = 0.01

    # Mask options.
    mask_feature_map: bool = False

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Object instance options.
    use_detections: bool = True
    num_preds_factor: float = 1.0
    min_visibility: float = 0.1

    # Feature extraction options.
    extractor_name: str = "dinov2_vitl14"
    grid_cell_size: float = 1.0
    max_num_queries: int = 1000000

    # Feature matching options.
    match_template_type: str = "cosine"
    match_top_n_templates: int = 5
    match_feat_matching_type: str = "cyclic_buddies"
    match_top_k_buddies: int = 300

    # PnP options.
    pnp_type: str = "opencv"
    pnp_ransac_iter: int = 1000
    pnp_required_ransac_conf: float = 0.99
    pnp_inlier_thresh: float = 10.0
    pnp_refine_lm: bool = True

    final_pose_type: str = "best_coarse"

    # Other options.
    save_estimates: bool = True
    vis_results: bool = True
    vis_corresp_top_n: int = 100
    vis_feat_map: bool = True
    vis_for_paper: bool = True
    debug: bool = True


def infer(opts: InferOpts) -> None:

    datasets_path = bop_config.datasets_path

    # Prepare a logger and a timer.
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    timer = misc_util.Timer(enabled=opts.debug)
    timer.start()

    # Load pre-generated detections saved in the BOP format.
    detections = {}
    if opts.use_detections:
        path = os.path.join(
            datasets_path,
            "detections",
            "cnos-fastsam",
            f"cnos-fastsam_{opts.object_dataset}-test.json",
        )
        detections = infer_pose_util.load_detections_in_bop_format(path)


    # Prepare feature extractor.
    extractor = feature_util.make_feature_extractor(opts.extractor_name)
    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    # Get IDs of objects to process.
    object_lids = opts.object_lids

    bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=opts.object_dataset)
    if object_lids is None:
        # If local (object) IDs are not specified, synthesize templates for all objects
        # in the specified dataset.
        object_lids = bop_model_props["obj_ids"]

    # Get properties of the test split of the specified dataset.
    bop_test_split_props = dataset_params.get_split_params(
        datasets_path=datasets_path,
        dataset_name=opts.object_dataset,
        split="test"
    )

    # Load BOP test targets
    test_targets_path = os.path.join(bop_test_split_props["base_path"], "test_targets_bop19.json")
    targets = inout.load_json(test_targets_path)

    scene_ids = dataset_params.get_present_scene_ids(bop_test_split_props)

    scene_im_ids = {}
    test_target_count = {}
    targets_per_obj = {}
    for target in targets:
        scene_im_ids.setdefault(target["scene_id"], set()).add(target["im_id"])
        key = (target["scene_id"], target["im_id"], target["obj_id"])
        test_target_count[key] = target["inst_count"]
        targets_per_obj.setdefault(target["obj_id"], list()).append(target)

    scene_gts = {}
    scene_gts_info = {}
    scene_cameras = {}
    for scene_id in scene_im_ids.keys():
        scene_cameras[scene_id] = data_util.load_chunk_cameras(bop_test_split_props["scene_camera_tpath"].format(scene_id=scene_id), bop_test_split_props["im_size"])
        scene_gts[scene_id] = data_util.load_chunk_gts(bop_test_split_props["scene_gt_tpath"].format(scene_id=scene_id),opts.object_dataset)
        scene_gts_info[scene_id] = json_util.load_json(
            bop_test_split_props["scene_gt_info_tpath"].format(scene_id=scene_id),
            keys_to_int=True,
        )

    # Create a renderer.
    renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
    renderer = renderer_builder.build(renderer_type=renderer_type, model_path=bop_model_props["model_tpath"])


    timer.elapsed("Time for setting up the stage")

    # results_final = []
    
    # Run inference for each specified object.
    for object_lid in object_lids:
        timer.start()

        # The output folder is named with slugified dataset path.
        version = opts.version
        if version == "":
            version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        signature = misc.slugify(opts.object_dataset) + "_{}".format(version)
        output_dir = os.path.join(
            bop_config.output_path, "inference", signature, str(object_lid)
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save parameters to a file.
        config_path = os.path.join(output_dir, "config.json")
        json_util.save_json(config_path, opts)

        # Create a pose evaluator.
        pose_evaluator = eval_util.EvaluatorPose([object_lid])

        # Load the object representation.
        logger.info(
            f"Loading representation for object {object_lid} from dataset {opts.object_dataset}..."
        )
        base_repre_dir = os.path.join(bop_config.output_path, "object_repre")
        repre_dir = repre_util.get_object_repre_dir_path(
            base_repre_dir, opts.repre_version, opts.object_dataset, object_lid
        )
        repre = repre_util.load_object_repre(
            repre_dir=repre_dir,
            tensor_device=device,
        )

        logger.info("Object representation loaded.")
        repre_np = repre_util.convert_object_repre_to_numpy(repre)


        # Get the object mesh and meta information.
        model_path = bop_model_props["model_tpath"].format(obj_id=object_lid)
        object_mesh = inout.load_ply(model_path)
        models_info = inout.load_json(bop_model_props["models_info_path"], keys_to_int=True)
        object_syms = bop_misc.get_symmetry_transformations(
            models_info[object_lid], max_sym_disc_step= 0.01
        )
        object_diameter = models_info[object_lid]["diameter"]

        max_vertices = 1000
        subsampled_vertices = np.random.permutation(object_mesh["pts"])[:max_vertices]

        timer.elapsed("Time for preparing object data")

       
        # Perform inference on each selected image.
        for item_info in targets_per_obj[object_lid]:
        # for scene_id in scene_ids:
            timer.start()


            # Chunk and image IDs in the original BOP dataset.
            bop_im_id = item_info["im_id"]
            bop_chunk_id = item_info["scene_id"]

            # if bop_chunk_id != 2 or bop_im_id != 322:
            #     continue

            # Get instance identifier if specified.
            inst_id = None
            if "inst_id" in item_info:
                inst_id = item_info["inst_id"]

            sample = data_util.prepare_sample(
                item_info,
                bop_test_split_props,
                scene_cameras,
                scene_gts,
                scene_gts_info
            )

            # Get object annotations.
            object_annos = []
            if sample.objects_anno is not None:
                all_object_annos = None
                if inst_id is not None:
                    all_object_annos = [sample.objects_anno[inst_id]]
                else:
                    all_object_annos = sample.objects_anno

                # Keep only GT annotations for sufficiently visible objects.
                for anno in all_object_annos:
                    if (
                        anno.lid == object_lid
                        and not np.isnan(anno.visibilities)
                        and anno.visibilities > opts.min_visibility
                    ):
                        object_annos.append(anno)

                # Continue if there are no sufficiently visible object annos.
                if len(object_annos) == 0:
                    continue

            # If test targets are specified use them to get the number of target instances.
            sample_key = (bop_chunk_id, bop_im_id, object_lid)
            if test_target_count is not None:
                # For test images.
                if sample_key not in test_target_count:
                    continue
                # Number of target instances
                num_target_insts = test_target_count[sample_key]

            else:
                num_target_insts = len(object_annos)

            # Skip this image if there are no test instances of the current object.
            if num_target_insts == 0:
                logger.info(f"Skipping image {bop_chunk_id}/{bop_im_id} because no GT.")
                continue

            msg = (
                f"Estimating pose of object {object_lid} in "
                f"scene_id {bop_chunk_id}, im_id {bop_im_id} "
                f"dataset {datasets_path}"
            )
            logging.log_heading(logger, msg, style=logging.BLUE_BOLD)

            # Camera parameters.
            orig_camera_c2w = sample.camera
            orig_image_size = (
                orig_camera_c2w.width,
                orig_camera_c2w.height,
            )

            # Get info about object instances for which we want to estimate pose.
            instances = infer_pose_util.get_instances_for_pose_estimation(
                bop_chunk_id=bop_chunk_id,
                bop_im_id=bop_im_id,
                obj_id=object_lid,
                use_detections=opts.use_detections,
                detections=detections,
                max_num_preds=int(opts.num_preds_factor * num_target_insts),
                gt_object_annos=object_annos,
                image_size=orig_image_size,
            )
            if len(instances) == 0:
                logger.info("No object instance, skipping.")
                continue

            if opts.mask_feature_map:
                if opts.crop:
                    grid_size = opts.crop_size
                else:
                    grid_size = orig_image_size
                grid_points = feature_util.generate_grid_points(
                    grid_size=grid_size,
                    cell_size=opts.grid_cell_size,
                )
                grid_points = grid_points.to(device)

            timer.elapsed("Time for preparing image data")

            # Estimate pose for each object instance.
            for inst_j, instance in enumerate(instances):
                times = {}

                if opts.use_detections:
                    # Add time for CNOS prediction.
                    pose_evaluator.detection_times[(bop_chunk_id, bop_im_id)] = (
                        instance["time"]
                    )
                    cnos_time = instance["time"]
                    logger.info(f"Time for segmentation: {cnos_time:.5f}s")

                    # Skip the prediction mask if it doesn't overlap with the ground truth.
                    if instance["gt_anno"] is not None:
                        mask_iou = eval_errors.mask_iou(
                            instance["input_mask_modal"],
                            instance["gt_anno"].masks_modal,
                        )
                        if mask_iou < 0.05:
                            continue

                    # Skip if the detection covers more than half of the image (e.g. itodd).
                    if (
                        instance["input_mask_modal"].sum()
                        > orig_image_size[0] * orig_image_size[1]
                    ):
                        continue
                else:
                    pose_evaluator.detection_times[(bop_chunk_id, bop_im_id)] = 0

                # Get the input image.
                orig_image_np_hwc = sample.image.astype(np.float32)/255.0

                # Get the modal mask and amodal bounding box of the instance.
                orig_mask_modal = instance["input_mask_modal"]
                orig_box_amodal = AlignedBox2f(
                    left=instance["input_box_amodal"][0],
                    top=instance["input_box_amodal"][1],
                    right=instance["input_box_amodal"][2],
                    bottom=instance["input_box_amodal"][3],
                )

                timer.start()

                # Optional cropping.
                if not opts.crop:
                    camera_c2w = orig_camera_c2w
                    image_np_hwc = orig_image_np_hwc
                    mask_modal = orig_mask_modal
                    box_amodal = orig_box_amodal
                else:
                    # Get box for cropping.
                    crop_box = misc_util.calc_crop_box(
                        box=orig_box_amodal,
                        make_square=True,
                    )

                    # Construct a virtual camera focused on the crop.
                    crop_camera_model_c2w = misc_util.construct_crop_camera(
                        box=crop_box,
                        camera_model_c2w=orig_camera_c2w,
                        viewport_size=opts.crop_size,
                        viewport_rel_pad=opts.crop_rel_pad,
                    )

                    # Map images to the virtual camera.
                    interpolation = (
                        cv2.INTER_AREA
                        if crop_box.width >= crop_camera_model_c2w.width
                        else cv2.INTER_LINEAR
                    )
                    image_np_hwc = warp_image(
                        src_camera=orig_camera_c2w,
                        dst_camera=crop_camera_model_c2w,
                        src_image=orig_image_np_hwc,
                        interpolation=interpolation,
                    )
                    mask_modal = warp_image(
                        src_camera=orig_camera_c2w,
                        dst_camera=crop_camera_model_c2w,
                        src_image=orig_mask_modal,
                        interpolation=cv2.INTER_NEAREST,
                    )

                    # Recalculate the object bounding box (it changed if we constructed the virtual camera).
                    ys, xs = mask_modal.nonzero()
                    box = np.array(misc_util.calc_2d_box(xs, ys))
                    box_amodal = AlignedBox2f(
                        left=box[0],
                        top=box[1],
                        right=box[2],
                        bottom=box[3],
                    )

                    # The virtual camera is becoming the main camera.
                    camera_c2w = crop_camera_model_c2w

                times["prep"] = timer.elapsed("Time for preparation")
                timer.start()

                # Extract feature map from the crop.
                image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
                image_tensor_bchw = image_tensor_chw.unsqueeze(0)
                extractor_output = extractor(image_tensor_bchw) # TODO: check the length and meaning of the feature maps
                feature_map_chw = extractor_output["feature_maps"][0]

                times["feat_extract"] = timer.elapsed("Time for feature extraction")

                if opts.mask_feature_map:
                    mask_modal_tensor = array_to_tensor(mask_modal).to(device)
                   

                timer.start()

                # Perform template matching.
                template_ids, template_scores = template_util.template_matching(
                    query_features=feature_map_chw,
                    object_repre=repre,
                    top_n_templates=opts.match_top_n_templates,
                    matching_type="cosine",
                    mask = mask_modal_tensor if opts.mask_feature_map else None,
                    grid_points = grid_points if opts.mask_feature_map else None,
                )
                times["template_matching"] = timer.elapsed("Time for template matching")

                # Normalize the template scores to probability
                template_scores = template_scores/sum(template_scores)
                template_ids = template_ids.cpu().numpy()
                template_scores = template_scores.cpu().numpy()

                # Infer poses from the matched templates.
                timer.start()
                scored_poses = []
                for template_rank, template_id in enumerate(template_ids):
                    camera_pose = repre.template_cameras_cam_from_model[template_id]

                    camera_K = np.zeros((3,3))
                    camera_K[0,0] = camera_pose.f[0]
                    camera_K[1,1] = camera_pose.f[1]
                    camera_K[0,2] = camera_pose.c[0]
                    camera_K[1,2] = camera_pose.c[1]
                    camera_K[2,2] = 1.0
          
                    scored_poses.append(
                        {
                            "template_id": int(template_id),
                            "template_score": float(template_scores[template_rank]),
                            "template_rank": int(template_rank),
                            "pose_template_mc": camera_pose.T_world_from_eye.tolist(),
                            "pose_template_cm": np.linalg.inv(camera_pose.T_world_from_eye).tolist(), #this is 100% true
                            "template_camera_K": camera_K.tolist(),
                        }
                    )

                times["pose_processing"] = timer.elapsed("Time for pose processing")

                # Store the results

                # store the query and its representation 
                base_query_dir = os.path.join(bop_config.output_path, "query_features")
                chunk_query_dir = os.path.join(base_query_dir, opts.object_dataset, str(bop_chunk_id))
                os.makedirs(chunk_query_dir, exist_ok=True)

                # make the directories for the query features, cropped projected mask and cropped projected images
                os.makedirs(os.path.join(chunk_query_dir, "scored_templates"), exist_ok=True)
                os.makedirs(os.path.join(chunk_query_dir, "virtual_cameras"), exist_ok=True)
                os.makedirs(os.path.join(chunk_query_dir, "features"), exist_ok=True)
                os.makedirs(os.path.join(chunk_query_dir, "masks"), exist_ok=True)
                os.makedirs(os.path.join(chunk_query_dir, "images"), exist_ok=True)

                # store the scored templates
                path_for_scores = os.path.join(chunk_query_dir, "scored_templates", f"{bop_im_id:06d}.json")
                with open(path_for_scores, 'w') as f:
                    json.dump(scored_poses, f, indent=4)

                # store the query feature map
                torch.save(feature_map_chw, os.path.join(chunk_query_dir, "features", f"{bop_im_id:06d}.pt"))

                # store the cropped projected mask
                mask_modal_tensor = array_to_tensor(mask_modal).to(device)
                query_points = feature_util.filter_points_by_mask(
                    grid_points, mask_modal_tensor
                )
                torch.save(query_points, os.path.join(chunk_query_dir, "masks", f"{bop_im_id:06d}.pt"))


                # store the cropped projected image
                image_pil = Image.fromarray((image_np_hwc * 255).astype(np.uint8))
                image_pil.save(os.path.join(chunk_query_dir, "images", f"{bop_im_id:06d}.png"))
                
                # store virtual camera info
                with open(os.path.join(chunk_query_dir, "virtual_cameras", f"{bop_im_id:06d}.json"), 'w') as f:
                    json.dump({
                        "camera": camera_c2w.T_world_from_eye.tolist(),
                        "K_c": [c.item() for c in camera_c2w.c],
                        "K_f": [f.item() for f in camera_c2w.f],
                        "cropped_img_size": [camera_c2w.width, camera_c2w.height],
                    }, f, indent=4)


        # TODO: this is all the data we need for the multiview optimization.

        # 1. we will generate the best consistent template
        # 2. use the best consistent template for correspondences
        # 3. use the correspondences for PnP
        # 4. use the PnP for pose refinement/featuremetric alignment
        # 5. somehow do everything in multiview    

        # Empty unused GPU cache variables.
        if device == "cuda":
            time_start = time.time()
            torch.cuda.empty_cache()
            gc.collect()
            time_end = time.time()
            logger.info(f"Garbage collection took {time_end - time_start} seconds.")

        # Save the pose estimates.          
        # if opts.save_estimates:
        #     results_path = os.path.join(output_dir, "estimated-poses.json")
        #     logger.info("Saving estimated poses to: {}".format(results_path))
        #     pose_evaluator.save_results_json(results_path)
        

def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(
        InferOpts
    )[0]
    infer(opts)


if __name__ == "__main__":
    main()
