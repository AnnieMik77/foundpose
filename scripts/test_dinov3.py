import torch
from transformers import AutoImageProcessor, AutoModel, DINOv3ViTImageProcessorFast
from transformers.image_utils import load_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

# make square
min_side = min(image.size)
left = (image.width - min_side) / 2
top = (image.height - min_side) / 2
right = (image.width + min_side) / 2
bottom = (image.height + min_side) / 2
image = image.crop((left, top, right, bottom))

# to torch
image = image.convert("RGB")
image = T.ToTensor()(image)

print(image.shape)  # (3, H, W)


# save image
plt.imshow(image.permute(1, 2, 0))
plt.axis("off")
plt.savefig("dinov3_original_image.png")

model = AutoModel.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    dtype=torch.float32,
    attn_implementation="sdpa"
)


normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
inputs = normalize(image).unsqueeze(0).to(model.device)  # (1, 3, H, W)

feature_type = 'patch'  # 'cls', 'reg', 'patch'

# forward pass
with torch.inference_mode():
    outputs = model(inputs, output_hidden_states=True)
    # layer_outputs = outputs.hidden_states[20]

           
    # if feature_type == 'cls':
    #     out = layer_outputs[:, 0, :]
    # elif feature_type == 'reg':
    #     out = layer_outputs[:, 1 : model.num_register_tokens + 1]
    # elif feature_type == 'patch':
    #     out = layer_outputs[:, 1 + model.config.num_register_tokens:, :]
    # else:
    #     raise NotImplementedError('')
    # if normalize:
    #     out /= out.norm(dim=-1, keepdim=True)
                

B, N, D = outputs.last_hidden_state.shape  # (1, 197, 384)
patch_tokens = outputs.last_hidden_state[:, 5:, :]  # drop CLS → (1, 196, 384)
patch_tokens = patch_tokens.squeeze(0)  # (196, 384)












pca = PCA(n_components=3)
features_pca = pca.fit_transform(patch_tokens.numpy())

pca_image = features_pca.reshape((30, 30, 3))

pca_image_normalized = np.zeros_like(pca_image)
for i in range(3):
    channel = pca_image[:, :, i]
    pca_image_normalized[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())

pca_pil_image = Image.fromarray((pca_image_normalized * 255).astype(np.uint8))

pca_pil_image_nearest = pca_pil_image.resize((224, 224), Image.Resampling.NEAREST)
pca_pil_image_bilinear = pca_pil_image.resize((224, 224), Image.Resampling.BILINEAR)
pca_pil_image_bicubic = pca_pil_image.resize((224, 224), Image.Resampling.BICUBIC)

# save PCA images
pca_pil_image_nearest.save("dinov3_pca_image_nearest.png")
pca_pil_image_bilinear.save("dinov3_pca_image_bilinear.png")
pca_pil_image_bicubic.save("dinov3_pca_image_bicubic.png")