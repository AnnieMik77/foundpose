import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from kornia.feature import DenseSIFTDescriptor
from kornia.color import rgb_to_grayscale
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


class RGBFeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Parse the model name.
        name_items = model_name.split("_")
        assert name_items[0] == "RGB"
        self.patch_size = int(name_items[1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Assuming images are in the shape (B, C, H, W)
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, C, H_out, W_out),
            where H_out = H // patch_size and W_out = W // patch_size.
        """
        B, C, H, W = images.shape
        p = self.patch_size

        # Ensure divisible dimensions
        assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size"

        # Use avg_pool2d to compute per-patch means efficiently
        # return images
        out = F.avg_pool2d(images, kernel_size=p, stride=p)
        result = {
            "feature_maps": out,
        }
        return result


class SIFTFeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Parse the model name. Example: "sift_stride=7_bins=4_binsize=4_pool=1_norm=1"
        # should be: bins + binsize = stride + 1
        name_items = model_name.split("_")
        assert name_items[0] == "sift"

        # Default parameters
        self.cell_stride = 7
        self.stride = 2
        self.bins = 2
        self.binsize = 7
        self.apply_norm = True
        self.pool = 1

        for item in name_items[1:]:
            name, value = item.split("=")
            if name == "stride":
                self.stride = int(value)
            elif name == "cell-stride":
                self.cell_stride = int(value)
            elif name == "bins":
                self.bins = int(value)
            elif name == "binsize":
                self.binsize = int(value)
            elif name == "norm":
                self.apply_norm = bool(int(value))
            elif name == "pool":
                self.pool = int(value)
        self.model = DenseSIFTDescriptor(
            num_spatial_bins=self.bins,
            spatial_bin_size=self.binsize,
            cell_stride=self.cell_stride,
            cell_padding=0,
            stride=self.stride,
            padding=0,
        ).to(device)
        # self.model = DenseSIFTDescriptor().to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Assuming images are in the shape (B, C, H, W)
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, C, H_out, W_out),
            where H_out = H // patch_size and W_out = W // patch_size.
        """
        B, C, H, W = images.shape
        p = self.pool

        # Ensure divisible dimensions
        assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size"

        images = images.to(device)
        images_gray = rgb_to_grayscale(images)
        with torch.no_grad():
            out = self.model(images_gray)
            if self.pool > 1:
                out = F.avg_pool2d(out, kernel_size=p, stride=p)
            # normalize to have mean 0 and std 1 per feature map
            if self.apply_norm:
                out = out - out.mean(dim=1, keepdim=True)
                out = out / (out.std(dim=1, keepdim=True) + 1e-6)

        result = {
            "feature_maps": out,
        }
        return result


#     # PATCH_SIZE_SIFT = 7  # Good for 420 size aka. DINOv2
#     PATCH_SIZE_SIFT = 8   # Good for 480 size aka. DINOv3

#     device = "cuda"

#     out_dir = Path("feature_vis_dir")
#     out_dir.mkdir(parents=True, exist_ok=True)

#     tpl_img_path = "view_base_cam_1626/template_img.png"
#     query_img_path = "view_base_cam_1626/query_img.png"

#     # read img and convert to grayscale
#     tpl_ts = pil_to_tensor(Image.open(tpl_img_path)).float()  # 0-255 range
#     query_ts = pil_to_tensor(Image.open(query_img_path)).float()  # 0-255 range
#     tpl_ts = tpl_ts.unsqueeze(0)  # (3, H, W) -> # (1, 3, H, W)
#     query_ts = query_ts.unsqueeze(0)  # (3, H, W) -> # (1, 3, H, W)

#     # # kornia uses 0-1 range for all color spaces/functions
#     # tpl_ts = rgb255_to_rgb(tpl_ts)
#     # query_ts = rgb255_to_rgb(query_ts)

#     # # to gray scale
#     # tpl_ts_gray = rgb_to_grayscale(tpl_ts)
#     # query_ts_gray = rgb_to_grayscale(query_ts)


# #     ############
# #     # SIFT
# #     dense_sift = DenseSIFTDescriptor().to(device, tpl_ts.dtype)
# #     with torch.no_grad():
# #         tpl_dense_sift = dense_sift(tpl_ts_gray)
# #         query_dense_sift = dense_sift(query_ts_gray)
# #         tpl_dense_sift_pooled = F.avg_pool2d(tpl_dense_sift, kernel_size=PATCH_SIZE_SIFT, stride=PATCH_SIZE_SIFT)
# #         query_dense_sift_pooled = F.avg_pool2d(query_dense_sift, kernel_size=PATCH_SIZE_SIFT, stride=PATCH_SIZE_SIFT)
# #         tpl_dense_sift_pooled = F.normalize(tpl_dense_sift_pooled, p=2, dim=1)  # avg pooling breaks normalization
# #         query_dense_sift_pooled = F.normalize(query_dense_sift_pooled, p=2, dim=1)  # avg pooling breaks normalization

# #     print(tpl_ts_gray.shape)
# #     print(tpl_dense_sift.shape)
# #     print(query_dense_sift.shape)
# #     print(tpl_dense_sift_pooled.shape)
# #     print(query_dense_sift_pooled.shape)


# #     ###############
# #     # COMPUTE RESIDUALS
# #     ###############

# #     # SIFT
# #     res_dense_sift = tpl_dense_sift - query_dense_sift  # (1,128,480,480)
# #     res_dense_sift_pooled = tpl_dense_sift_pooled - query_dense_sift_pooled  # (1,128,60,60)

# #     ##########
# #     # VIS
# #     # NOTE
# #     # to_pil_image implicitely converts floating point
# #     # tensors to (255*ts).to(torch.uint8)
# #     ##########

# #     # SIFT
# #     res_norm_vis = res_dense_sift.norm(dim=1)[0]
# #     res_norm_vis /= res_norm_vis.max()
# #     res_dense_sift_pooled = res_dense_sift_pooled.norm(dim=1)[0]
# #     res_dense_sift_pooled /= res_dense_sift_pooled.max()
# #     to_pil_image(res_norm_vis).save(out_dir / "res_dense_sift.png")
# #     to_pil_image(res_dense_sift_pooled).save(out_dir / "res_dense_sift_pooled.png")


# # if __name__ == "__main__":
# #     device = "cuda"
# #     extractor = SIFT(model_name="sift_8").to(device)
# #     image = torch.randn(1, 3, 480, 480).to(device)
# #     output = extractor(image)
# #     print(output["feature_maps"].shape) # should be (1, 128, 60, 60)
