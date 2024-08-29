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

import re
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPImageProcessor
from PIL import Image
import requests
import torch.nn.functional as F
from transformers import AutoProcessor, Pix2StructVisionModel, Pix2StructProcessor, Pix2StructForConditionalGeneration

cfg={
    "crop_size": 256,
    "do_center_crop": True,
    "do_normalize": True,
    "do_resize": True,
    "feature_extractor_type": "CLIPFeatureExtractor",
    "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
    ],
    "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
    ],
    "resample": 3,
    "size": 256
}

'''
Pixel2Struct-Large Model (pretrained version)
'''
class Pix2StructLargeVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.do_resize = args.do_resize
        self.de_normalize = args.de_normalize # de-normalize the input image and perform preprocessing with pix2struct processor
        self.select_layer = args.mm_vision_select_layer # NOTE: not implemented yet, this parameter has no effect
        self.input_image_size = args.input_image_size
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.freeze_vision = args.freeze_vision

        self.args = args
        if not self.is_loaded:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            return
        whole_model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-large")
        self.vision_tower = whole_model.encoder
        self.pix2struct_processor = AutoProcessor.from_pretrained("google/pix2struct-large")
        self.pix2struct_processor.image_processor.is_vqa = False

        self.image_processor = CLIPImageProcessor(**cfg)
        if self.input_image_size is not None:
            self.image_processor.size=self.input_image_size
            self.image_processor.crop_size={
                'height':self.input_image_size,
                'width': self.input_image_size
            }

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)
        
        self.image_mean = torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1)
        
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer] # [bs, n, c], cls at idx=0
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images):

        if self.de_normalize:
            mean = self.image_mean.clone().view(1, 3, 1, 1).to(dtype=images.dtype, device=images.device)
            std = self.image_std.clone().view(1, 3, 1, 1).to(dtype=images.dtype, device=images.device)
            x = (images * std + mean) * 255.0
            x = self.pix2struct_processor(images=x.float(), return_tensors="pt")

        image_features = self.vision_tower(**(x.to(device=self.device, dtype=self.dtype))).last_hidden_state
        bs, n, c = image_features.shape
        image_features  = image_features[:, :2025, :] # HARD CODE
        
        if self.do_resize:
            image_features = image_features.transpose(1,2).reshape(bs, c, 45, 45) # HARD CODE
            image_features = F.interpolate(image_features.float(), size=(32, 32), mode='bilinear', align_corners=True).to(dtype=image_features.dtype) # HARD CODE
            return image_features
        else:
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
        return self.vision_tower.config

    @property
    def hidden_size(self):
        # Hard code
        hidden_dim = 1536
        return hidden_dim

    @property
    def num_patches(self):
        return self.config['num_patches']
