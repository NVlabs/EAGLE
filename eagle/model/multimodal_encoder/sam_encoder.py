# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

import transformers
from transformers import SamProcessor
from transformers import SamModel, SamVisionConfig, SamVisionConfig
from transformers import SamImageProcessor
from PIL import Image


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->Sam
class SamLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ShortSamVisionNeck(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = SamLayerNorm(config.output_channels, data_format="channels_first")

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = hidden_states.permute(0,2,3,1)
        return hidden_states


class SAMVisionTower(nn.Module):
    def __init__(self, vision_tower, args):
        super().__init__()

        self.args = args
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.input_image_size = args.input_image_size
        self.pixel_shuffle = getattr(args, 'add_pixel_shuffle', False)

        self.freeze = args.freeze_vision

        self.load_model()

    def load_model(self):
        if self.is_loaded:
            return

        self.image_processor= SamProcessor.from_pretrained("facebook/sam-vit-large")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-large").vision_encoder
        sam_model.neck = ShortSamVisionNeck(sam_model.config)
        self.sam_model_config = sam_model.config
        self.image_processor.preprocess = self.image_processor.__call__
        self.image_processor.image_mean = [0.485,0.456,0.406]
        self.vision_tower = sam_model
        
        if self.freeze:
            self.vision_tower.requires_grad_(False)
            
        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device).unsqueeze(0))
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device)).last_hidden_state.flatten(start_dim=1, end_dim=2).to(device=self.device)

        if self.pixel_shuffle:
            b, n, c = image_features.shape
            h = w = int(n ** 0.5)
            image_features = image_features.transpose(1,2).reshape(b, c, h, w) 
            image_features = nn.functional.pixel_unshuffle(image_features, 2)

        return image_features
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self.sam_model_config

    @property
    def hidden_size(self):
        # Hard code
        if self.pixel_shuffle:
            hidden_size = 256 * 4
        else:
            hidden_size = 256
        return hidden_size

    @property
    def num_patches(self):
        return self.config.num_patches
