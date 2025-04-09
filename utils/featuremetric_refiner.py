from pixloc.pixlib.models.classic_optimizer import ClassicOptimizer
from pixloc.pixlib.geometry import Camera, Pose
from typing import Tuple
from torch import Tensor
from utils.structs import PinholePlaneCameraModel, ObjectPose
import numpy as np
import torch
from utils import misc

def refine(
        template_vertices_ref: Tensor,
        template_masked_features_ref: Tensor,
        feature_map_chw_proj_ref: Tensor,
        initial_pose: ObjectPose,
        camera_c2w: PinholePlaneCameraModel,
        image_size: Tuple[int, int],
      ) -> Tuple[Pose, Tensor]:
    """
    Refine the pose using the ClassicOptimizer.
    Args:
        template_vertices_ref: Template vertices. Shape (B, N, 3).
        template_masked_features_ref: Template masked features. Shape (B, N, C).
        feature_map_chw_proj_ref: Query feature map. Shape (B, C, H, W). Heights and widths are the same as image size.
        initial_pose: Initial pose.
        camera_c2w: Camera model.
        image_size: Image size.
    Returns:
        Tuple[ObjectPose, Tensor]: Optimized pose and failure flag.
    """
    device = template_vertices_ref.device

    camera_intrinsic = torch.tensor([
        *image_size,
        camera_c2w.f[0],
        camera_c2w.f[1],
        camera_c2w.c[0],
        camera_c2w.c[1]
        ], dtype=torch.float32)
    camera_model = Camera(data=camera_intrinsic).to(device)

    initial_pose = misc.get_rigid_matrix(initial_pose)
    initial_pose_tensor = torch.tensor(initial_pose, dtype=torch.float32)
    initial_pose_tensor = Pose.from_4x4mat(initial_pose_tensor.unsqueeze(0)).to(device)

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
    optimizer = ClassicOptimizer(conf)
    T_pose, failed = optimizer.run(
        template_vertices_ref,
        template_masked_features_ref,
        feature_map_chw_proj_ref,
        initial_pose_tensor, 
        camera_model)
    
    optimized_pose = ObjectPose(
                            R=T_pose.R.squeeze().cpu(),
                            t=T_pose.t.squeeze().cpu()
                        )
    
    return optimized_pose, failed