import torch
import torch.nn as nn
import re
# from llava.model.multimodal_projector.deformable_resampler import DeformableResampler

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, fpn_input_dim=[], **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    # resampler_match = re.match(r'^deformable-resampler-l(\d+)d(\d+)p(\d+)', projector_type)
    # if resampler_match:
    #     use_fpn = "fpn" in projector_type or len(fpn_input_dim) > 0
    #     layer_num = int(resampler_match.group(1))
    #     embed_dim = int(resampler_match.group(2))
    #     sample_point = int(resampler_match.group(3))
    #     if len(fpn_input_dim) > 0:
    #         fpn_type = 'multi-level'
    #     else:
    #         fpn_type = 'simple'

    #     return DeformableResampler(input_dimension=config.mm_hidden_size,
    #                                output_dimension=config.hidden_size,
    #                                query_number=config.mm_projector_query_number,
    #                                num_layers=layer_num,
    #                                num_heads=8,      
    #                                feedforward_dims=2048,
    #                                embed_dims=embed_dim, 
    #                                num_points=sample_point,
    #                                direct_projection=True,
    #                                use_fpn=use_fpn,
    #                                fpn_config=dict(
    #                                    fpn_type=fpn_type,
    #                                    in_channels=fpn_input_dim))

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
