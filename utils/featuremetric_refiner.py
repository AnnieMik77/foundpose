from pixloc.pixlib.models.classic_optimizer import ClassicOptimizer
from pixloc.pixlib.geometry import Camera, Pose
from typing import List, Tuple
from torch import Tensor
from utils.structs import PinholePlaneCameraModel, ObjectPose
import numpy as np
import torch
from utils import misc


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

