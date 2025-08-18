import torch
from torch import Tensor
import numpy as np

from pixloc.pixlib.models.classic_optimizer import ClassicOptimizer
from pixloc.pixlib.models.multiview_optimizer import ClassicMultiviewOptimizer
from pixloc.pixlib.geometry import Camera, Pose
from utils import misc
from utils.structs import PinholePlaneCameraModel, ObjectPose
from utils.feature_repre import FeatureRepre
from typing import List, Tuple

def templates_to_tensor(templates: List) -> Tuple[Tensor, Tensor]:
    # Get data from templates
    template_vertices_ref = [template.vertices for template in templates]
    template_masked_features_ref = [template.masked_features for template in templates]

    mask = []
    max_len = max([vertices.shape[0] for vertices in template_vertices_ref])
    for i, (vertices, features) in enumerate(zip(template_vertices_ref, template_masked_features_ref)):
        # Pad the vertices and features to the same length
        pad_len = max_len - vertices.shape[0]
        if pad_len > 0:
            template_vertices_ref[i] = torch.nn.functional.pad(vertices, (0, 0, 0, pad_len), value=0)
            template_masked_features_ref[i] = torch.nn.functional.pad(features, (0, 0, 0, pad_len), value=0)
            mask.append(
                torch.tensor([True] * vertices.shape[0] + [False] * pad_len, dtype=torch.bool)
            )
        else:
            template_vertices_ref[i] = vertices[:max_len]
            template_masked_features_ref[i] = features[:max_len]
            mask.append(torch.ones(max_len, dtype=torch.bool))


    template_vertices_ref = torch.stack(template_vertices_ref, dim=0)
    template_masked_features_ref = torch.stack(template_masked_features_ref, dim=0)
    mask = torch.stack(mask, dim=0).to(device=template_vertices_ref.device)

    return template_vertices_ref, template_masked_features_ref, mask

def queries_to_tensor(queries: List, img_size: Tuple) -> Tuple[Tensor, Tensor]:
    feature_map_chw_proj_ref = []
    for query in queries:
        feature_map = query.features.T.reshape(-1,30,30).unsqueeze(0)
        feature_map = torch.nn.functional.interpolate(feature_map,(img_size[0], img_size[1]), mode='bilinear', align_corners=True)
        feature_map_chw_proj_ref.append(feature_map.squeeze(0))
    feature_map_chw_proj_ref = torch.stack(feature_map_chw_proj_ref, dim=0)
    return feature_map_chw_proj_ref

def refine_fp_wrapper(
        initial_pose_m2c: ObjectPose,
        template_vertices_ref: Tensor,
        template_masked_features_ref: Tensor,
        feature_map_chw_proj_ref: Tensor,
        camera_c2w: PinholePlaneCameraModel,
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create initial pose
    initial_pose_m2c = misc.get_rigid_matrix(initial_pose_m2c)
    initial_pose_m2c = torch.tensor(initial_pose_m2c, dtype=torch.float32)
    initial_pose_m2c = Pose.from_4x4mat(initial_pose_m2c.unsqueeze(0)).to(device)

    # Create an instance of Camera
    camera_intrinsic = torch.tensor([
        camera_c2w.width,
        camera_c2w.height,
        camera_c2w.f[0],
        camera_c2w.f[1],
        camera_c2w.c[0],
        camera_c2w.c[1]
        ], dtype=torch.float32).unsqueeze(0)
    camera_model = Camera(data=camera_intrinsic).to(device)

    # Refine
    optimized_pose, failed = refine(
        template_vertices_ref = template_vertices_ref,
        template_masked_features_ref = template_masked_features_ref,
        feature_map_chw_proj_ref = feature_map_chw_proj_ref,
        initial_pose_m2c = initial_pose_m2c,
        camera = camera_model,
    )

    # Convert the optimized pose to ObjectPose
    optimized_pose = ObjectPose(
                            R=optimized_pose.R.squeeze().detach().cpu(),
                            t=optimized_pose.t.squeeze().detach().cpu()
                        )

    return optimized_pose, failed

def refine_wrapper(
        initial_pose_m2c: ObjectPose,
        templates: List[FeatureRepre],
        queries: List[FeatureRepre],
        cameras_c2w: List[PinholePlaneCameraModel],
):
    assert len(templates) == 1, "Refine wrapper only supports single template for now"
    assert len(queries) == 1, "Refine wrapper only supports single query for now"
    assert len(cameras_c2w) == 1, "Refine wrapper only supports single camera for now"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    camera_c2w = cameras_c2w[0]
    img_size = (camera_c2w.width, camera_c2w.height)

    # Get data from templates
    template_vertices_ref, template_masked_features_ref, _ = templates_to_tensor([templates[0]])

    # Get data from queries (need to upscale the feature map to the image size), remove grad
    query_features = queries_to_tensor(queries, (img_size[0], img_size[1]))
    query_features = query_features.detach().requires_grad_(False)

    # Create initial pose
    initial_pose_m2c = misc.get_rigid_matrix(initial_pose_m2c)
    initial_pose_m2c = torch.tensor(initial_pose_m2c, dtype=torch.float32)
    initial_pose_m2c = Pose.from_4x4mat(initial_pose_m2c.unsqueeze(0)).to(device)

    # Create an instance of Camera
    camera_intrinsic = torch.tensor([
        *img_size,
        camera_c2w.f[0],
        camera_c2w.f[1],
        camera_c2w.c[0],
        camera_c2w.c[1]
        ], dtype=torch.float32)
    camera_model = Camera(data=camera_intrinsic).to(device)

    # Refine
    optimized_pose, failed = refine(
        template_vertices_ref = template_vertices_ref,
        template_masked_features_ref = template_masked_features_ref,
        feature_map_chw_proj_ref = query_features,
        initial_pose_m2c = initial_pose_m2c,
        camera = camera_model,
    )

    # Convert the optimized pose to ObjectPose
    optimized_pose = ObjectPose(
                            R=optimized_pose.R.squeeze().detach().cpu(),
                            t=optimized_pose.t.squeeze().detach().cpu()
                        )

    return optimized_pose, failed

def refine(
        template_vertices_ref: Tensor,
        template_masked_features_ref: Tensor,
        feature_map_chw_proj_ref: Tensor,
        initial_pose_m2c: Pose,
        camera: Camera,
      ) -> Tuple[Pose, Tensor]:
    """
    Refine the pose using the ClassicOptimizer.
    Args:
        template_vertices_ref: Template vertices. Shape (B, N, 3).
        template_masked_features_ref: Template masked features. Shape (B, N, C).
        feature_map_chw_proj_ref: Query feature map. Shape (B, C, H, W). Heights and widths are the same as image size.
        initial_pose_m2c: Initial pose.
        camera: Camera model. Only intrinsics matter. Shape (B, 4, 4).
    Returns:
        Tuple[Pose, Tensor]: Optimized pose and failure flag.
    """

    # Create an instance of ClassicOptimizer
    conf = {
        "num_iters": 30,
        "lambda_": 1e-2,
        "lambda_max": 1e4,
        "normalize_features": True,
        "jacobi_scaling": False,
        "interpolation": dict(
            mode='linear',
            pad=4,
        ),
        "loss_fn": "scaled_barron(-5, 0.5)",
    }

    # Optimize the pose
    optimizer = ClassicOptimizer(conf)
    T_pose, failed = optimizer.run(
        p3D = template_vertices_ref,
        F_ref = template_masked_features_ref,
        F_query = feature_map_chw_proj_ref,
        T_init = initial_pose_m2c, 
        camera = camera)
    
    # TODO: Check if the optimization failed
    if failed:
        raise ValueError("Refinement failed")
    
    return T_pose, failed



def refine_multiview_wrapper(
        initial_pose_m2w: ObjectPose,
        templates: List[FeatureRepre],
        queries: List[FeatureRepre],
        cameras_c2w: List[PinholePlaneCameraModel], 
        **kwargs
    ) -> Tuple[np.array, bool]:
    """
    Refine the pose using the ClassicOptimizer.
    Args:
        initial_pose_m2w: Initial obj pose in world coordinates.
        templates: List of templates. Each template contains vertices and features.
        queries: List of queries. Each query contains features.
        cameras_c2w: List of camera models.
    Returns:
        Tuple[np.array, bool]: Optimized pose and failure flag.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assume all cameras have the same size (for query upscaling)
    img_size = (cameras_c2w[0].width,cameras_c2w[0].height)

    # Get data from templates
    template_vertices_ref, template_masked_features_ref, mask = templates_to_tensor(templates)

    # Get data from queries (need to upscale the feature map to the image size), remove grad
    feature_map_chw_proj_ref = queries_to_tensor(queries, img_size)
    feature_map_chw_proj_ref = feature_map_chw_proj_ref.detach().requires_grad_(False)

    # Convert initial pose from ObjectPose to Pose type
    initial_pose_m2w = misc.get_rigid_matrix(initial_pose_m2w)
    initial_pose_m2w = Pose.from_4x4mat(initial_pose_m2w).to(device).to(torch.float32)

    # Get camera data
    cam_intrinsics = []
    cam_poses = []
    for camera_c2w in cameras_c2w:
        camera_intrinsic = torch.tensor([
            *img_size,
            camera_c2w.f[0],
            camera_c2w.f[1],
            camera_c2w.c[0],
            camera_c2w.c[1]
            ], dtype=torch.float32).to(device)
        cam_intrinsics.append(camera_intrinsic)

        camera_pose_c2w = camera_c2w.T_world_from_eye
        camera_pose_c2w = torch.tensor(camera_pose_c2w, dtype=torch.float32).to(device)
        cam_poses.append(torch.linalg.inv(camera_pose_c2w))

    cam_intrinsics = Camera(data=torch.stack(cam_intrinsics)).to(device)
    cam_poses_w2c = Pose.from_4x4mat(torch.stack(cam_poses))

    # Refine
    refined_m2w, failed, cost_seq = refine_multiview(
        template_vertices_ref = template_vertices_ref,
        template_masked_features_ref = template_masked_features_ref,
        mask = mask,
        feature_map_chw_proj_ref = feature_map_chw_proj_ref,
        initial_pose_m2w = initial_pose_m2w,
        cameras = cam_intrinsics,
        poses_w2c = cam_poses_w2c,
        **kwargs
    )

    for s in cost_seq:
        print(s)

    # Get the optimized pose
    optimized_pose_m2w = ObjectPose(
                            R=refined_m2w.R.squeeze().detach().cpu(),
                            t=refined_m2w.t.detach().cpu()
                        )
    
    return optimized_pose_m2w, failed, cost_seq
        
def refine_multiview(
        template_vertices_ref: Tensor,
        template_masked_features_ref: Tensor,
        mask: Tensor,
        feature_map_chw_proj_ref: Tensor,
        initial_pose_m2w: Pose,
        cameras: Camera,
        poses_w2c: Pose,
        num_iters: int = 30,
        **kwargs
      ) -> Tuple[Pose, Tensor]:
    """
    Refine the pose using the ClassicOptimizer.
    Args:
        template_vertices_ref: Template vertices. list of shapes (M, N, 3).
        template_masked_features_ref: Template masked features. list of shapes (M, N, C).
        mask: Mask for valid template vertices. Shape (M, N).
        feature_map_chw_proj_ref: Query feature map. Shape (M, C, H, W). Heights and widths are the same as image size.
        initial_pose: Initial pose in world coordinates. Shape (4, 4).
        cameras: Camera models, len=M, containing camera intrinsics.
        poses_w2c: Camera poses in world. Shape (M, 4, 4).
    Returns:
        Tuple[Pose, Tensor]: Optimized pose and failure flag.
    """
    # Create an instance of ClassicOptimizer
    print(num_iters)
    conf = {
        "num_iters": num_iters,
        "lambda_": 1e-2,
        "lambda_max": 1e4,
        "normalize_features": True,
        "jacobi_scaling": False,
        "interpolation": dict(
            mode='linear',
            pad=4,
        ),
        "loss_fn": "scaled_barron(-5, 0.5)",
    }
    optimizer = ClassicMultiviewOptimizer(conf)
    optimizer.eval()

    # Run the optimizer
    T_pose, failed, cost_seq = optimizer.run(
        p3D=template_vertices_ref,
        F_ref=template_masked_features_ref,
        F_query=feature_map_chw_proj_ref,
        T_init_wo=initial_pose_m2w,
        T_cw=poses_w2c,
        cameras=cameras,
        mask=mask
        )
     
    # TODO: handle the case when the optimizer fails
    if failed:
        raise ValueError("Refinement failed")
    return T_pose, failed