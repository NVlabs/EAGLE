import os
from .clip_encoder import CLIPVisionTower
from .multi_backbone_channel_concatenation_encoder import MultiBackboneChannelConcatenationVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if "clip" in vision_tower and vision_tower.startswith("openai"):
        is_absolute_path_exists = os.path.exists(vision_tower)
        if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)       
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    
    elif ";" in vision_tower:
        return MultiBackboneChannelConcatenationVisionTower(vision_tower, args=vision_tower_cfg)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
