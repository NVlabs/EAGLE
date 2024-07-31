import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

import math
import torch
import torch.nn.functional as F
from typing import List, Optional


def forward_embeddings(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    batch_size = pixel_values.shape[0]
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    position_embeddings = self.position_embedding(self.position_ids)

    if position_embeddings.shape[1]!=embeddings.shape[1]:
        position_embeddings=resample_pos_embed(position_embeddings,embeddings.shape[1])

    embeddings = embeddings + position_embeddings
    return embeddings


def resample_pos_embed(
        posemb,
        new_size: int,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    new_size=[int(math.sqrt(new_size-num_prefix_tokens)),int(math.sqrt(new_size-num_prefix_tokens))]
    num_pos_tokens = posemb.shape[1] - num_prefix_tokens
    old_size = int(math.sqrt(num_pos_tokens))
    bs=posemb.shape[0]

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:,:num_prefix_tokens], posemb[:,num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(bs, old_size, old_size, -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(bs, -1, embed_dim)
    posemb = posemb.to(dtype=orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb],1)

    if not torch.jit.is_scripting() and verbose:
        print(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.freeze_vision=args.freeze_vision
        self.input_image_size=args.input_image_size
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')


        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)


    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        # checkpointing for clip
        self.vision_tower.vision_model.encoder.gradient_checkpointing =True



        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        cls_=self.vision_tower.vision_model.embeddings
        bound_method = forward_embeddings.__get__(cls_, cls_.__class__)
        setattr(cls_, 'forward', bound_method)

        if self.input_image_size is not None:
            self.image_processor.size=self.input_image_size
            self.image_processor.crop_size={
                'height':self.input_image_size,
                'width': self.input_image_size
            }

        self.is_loaded = True

    def feature_select(self, image_forward_outs):

        if self.select_layer>100:
            n_layer=len(image_forward_outs.hidden_states)-1
            mod_index = max(n_layer // (self.select_layer//100),1)
            image_features=[]
            for i in range(n_layer):
                if  (i+1)% mod_index==0:
                    image_features.append(image_forward_outs.hidden_states[i+1])
            image_features=torch.cat(image_features,-1)
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if self.freeze_vision:
            with torch.no_grad():
                image_features = self._forward_images(images)
        else:
            image_features = self._forward_images(images)

        return image_features

    def _forward_images(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device


    @property
    def num_attention_heads(self):
        return self.config.num_attention_heads
    @property
    def num_layers(self):
        return self.config.num_hidden_layers
    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2