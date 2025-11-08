
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DINOv3FeatureExtractor(nn.Module):
    REPO_DIR = (Path(os.path.dirname(__file__)).parent.parent.parent.parent / 'dinov3').resolve()

    def __init__(self, model_name: str = 'dinov3-vitl16-pretrain-lvd1689m'):
        super().__init__()
        #self.processor = AutoImageProcessor.from_pretrained(f"facebook/{model_name}")
        self.model = AutoModel.from_pretrained(f"facebook/{model_name}")

        self.model.eval()
        self.transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, images, layer=22, feature_type='cls', normalize=True):
        if images.shape[-1] != 480 or images.shape[-2] != 480:
            raise ValueError(f'Should be 480x480! Found {images.shape[-2]}x{images.shape[-1]}')
        
        with torch.inference_mode():
            #inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
            inputs = self.transform(images).to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(inputs, output_hidden_states=True)
                layer_outputs = outputs.hidden_states[layer]

            if feature_type == 'cls':
                out = layer_outputs[:, 0, :]
            elif feature_type == 'reg':
                out = layer_outputs[:, 1 : self.model.num_register_tokens + 1]
            elif feature_type == 'patch':
                out = layer_outputs[:, 1 + self.model.config.num_register_tokens:, :]
            else:
                raise NotImplementedError('')
            if normalize:
                out /= out.norm(dim=-1, keepdim=True)
                
        return out