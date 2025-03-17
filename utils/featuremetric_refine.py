from cv2 import Rodrigues
import torch
import numpy as np
from utils.feature_util import sample_feature_map_at_points
from scipy.optimize import least_squares
import torchmin


import numpy as np
from cv2 import Rodrigues

def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h

def rot_trans_to_matrix(R,t):
    # Reshape rotation vector into 3x3 matrix
    R = np.array(R).reshape(3, 3)
    t = np.array(t).reshape(3, 1)

    # Create 4x4 homogeneous transformation matrix
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()
    return pose

def rvec_tvec_to_matrix(rvec,tvec):
    R = Rodrigues(rvec)[0]
    tvec = tvec.squeeze()

    pose = rot_trans_to_matrix(R,tvec)
    return pose

def axis_angle_to_matrix(axis_angle: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
    cross_product_matrix = torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        identity.expand(cross_product_matrix.shape)
        + torch.sinc(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )

def rodrigues(log_rot, eps=0.0001):
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = axis_angle_to_matrix(log_rot)

    return R, rot_angles, skews, skews_square


def matrix_to_rvec_tvec(pose):
    R = pose[:3,:3]
    t = pose[:3,3]

    rvec,_ = Rodrigues(R)
    rvec = rvec.reshape(3)
    return rvec,t

def camera_intrinsics_to_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def camera_matrix_to_intrinsics(K):
    return [K[0,0], K[1,1]], [K[0,2], K[1,2]]

def camera_to_img(camera_K, points):
    """
    Projects 3D points in camera coordinates to 2D image coordinates
    """
    points = points / (points[2] + 1e-8)

    points_in_img = camera_K @ points
    return points_in_img[:2]


class OptimizationTracker:
    def __init__(self):
        self.input_poses = []  # Store x values
        self.residuals = []  # Store residuals
        self.template_masked_features = None
        self.template_feat_to_vertex_mapping = None
        self.template_vertices = None
        self.query_features = None
        self.camera_K = None

    def get_residuals(self, pose, template_masked_features, template_feat_to_vertex_mapping, template_vertices, query_features, camera_K):
        """
        For a given pose, computes the residuals between the template and query features

        Args:
        - pose: 6D vector representing the rotation and translation
        - template: template object
        - query: query object
        - camera_K: virtual camera intrinsics

        Returns:
        - feature_diffs: N x D array of feature differences
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 0. Pose to rotation and translation
        pose = pose.reshape(6)
        rvec = pose[:3]
        t = pose[3:]
        R = Rodrigues(np.array(rvec))[0]
        self.input_poses.append(rot_trans_to_matrix(R, t))

        # pose to gpu
        R = torch.tensor(R).to(device).float()
        t = torch.tensor(t).to(device).float()

        # virtual camera intrinsics to gpu
        camera_K = torch.tensor(camera_K).to(device).float()

        feature_diffs = []

        # 1. Features in template
        p_i = template_masked_features

        # 2. Points from object to virtual camera coordinates
        x_i = template_vertices[template_feat_to_vertex_mapping].T
        pt_in_camera_coords = R@x_i + t.repeat(x_i.shape[1], 1).T

        # 3. Points from camera coordinates to image coordinates
        pt_in_img_space = camera_to_img(camera_K, pt_in_camera_coords)

        # 4. Get the feature map at the point
        feature_at_projected_pt = sample_feature_map_at_points(
            query_features, pt_in_img_space.T, (420,420)
            ).squeeze()

        # 6. Calculate the difference 
        feature_diff = p_i - feature_at_projected_pt

        feature_diffs = feature_diff.flatten()

        return feature_diffs.detach().cpu().numpy()
    
    def get_residuals_torch(self,pose):
        """
        For a given pose, computes the residuals between the template and query features

        Args:
        - pose: 6D vector representing the rotation and translation
        - template: template object
        - query: query object
        - camera_K: virtual camera intrinsics

        Returns:
        - feature_diffs: N x D array of feature differences
        """
        template_masked_features = self.template_masked_features
        template_feat_to_vertex_mapping = self.template_feat_to_vertex_mapping
        template_vertices = self.template_vertices
        query_features = self.query_features
        camera_K = self.camera_K

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 0. Pose to rotation and translation
        rvec = pose[:3]
        t = pose[3:]
        # R1 = Rodrigues(rvec.detach().cpu().numpy())[0].to(device).float()
        R,_,_,_ = rodrigues(rvec.reshape(1,3))
        R = R.squeeze()
        # assert R1 == R

        # pose to gpu
        t = t.float()

        # virtual camera intrinsics to gpu
        camera_K = torch.tensor(camera_K).to(device).float()

        feature_diffs = []

        # 1. Features in template
        p_i = template_masked_features

        # 2. Points from object to virtual camera coordinates
        x_i = template_vertices[template_feat_to_vertex_mapping].T.squeeze()
        pt_in_camera_coords = R@x_i + t.repeat(x_i.shape[1], 1).T

        # 3. Points from camera coordinates to image coordinates
        pt_in_img_space = camera_to_img(camera_K, pt_in_camera_coords) # pi

        # 4. Get the feature map at the point #
        feature_at_projected_pt = sample_feature_map_at_points(
            query_features, pt_in_img_space.T, (420,420)
            ).squeeze()

        # 6. Calculate the difference 
        feature_diff = p_i - feature_at_projected_pt 
        feature_diffs = feature_diff.flatten()
        return feature_diffs
    
def minimize_with_scipy(initial_pose, template_masked_features, template_feat_to_vertex_mapping, template_vertices, query_features, virtual_camera_K, max_iter=100, verbose=0):
    tracker = OptimizationTracker()

    # pose to vector
    rvec,tvec = matrix_to_rvec_tvec(initial_pose)
    rt_vector = np.concatenate([rvec, tvec])

    result = least_squares(
        fun=tracker.get_residuals,
        x0=rt_vector.squeeze(),
        args=(template_masked_features, template_feat_to_vertex_mapping, template_vertices, query_features, virtual_camera_K),
        method="trf",
        loss="huber",
        max_nfev=max_iter,
        verbose=verbose
    )

    pose_vector = result.x
    final_pose = rvec_tvec_to_matrix(pose_vector[:3], pose_vector[3:])

    return final_pose, tracker.input_poses, tracker.residuals

def minimize_with_torchmin(initial_pose, template_masked_features, template_feat_to_vertex_mapping, template_vertices, query_features, virtual_camera_K, max_iter=100, verbose=0):
    tracker = OptimizationTracker()
    tracker.template_masked_features = template_masked_features
    tracker.template_feat_to_vertex_mapping = template_feat_to_vertex_mapping
    tracker.template_vertices = template_vertices
    tracker.query_features = query_features
    tracker.camera_K = virtual_camera_K

    # pose to vector
    rvec,tvec = matrix_to_rvec_tvec(initial_pose)
    rt_vector = np.concatenate([rvec, tvec])
    rt_vector = torch.tensor(rt_vector).to("cuda").float().squeeze()
    rt_vector.requires_grad = True


    result = torchmin.least_squares(
        fun=tracker.get_residuals_torch,
        x0=rt_vector,
        method="trf",
        max_nfev=max_iter,
        verbose=verbose
    )

    pose_vector = result.x
    final_pose = rvec_tvec_to_matrix(pose_vector[:3], pose_vector[3:])

    return final_pose, tracker.input_poses, tracker.residuals
