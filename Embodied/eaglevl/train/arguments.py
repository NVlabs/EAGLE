# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    lambda_box: float = field(default=0.1, metadata={'help': 'Canonical bbox GIoU weight; 0 disables the auxiliary loss.'})
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    save_every_n_hours: int = field(
        default=4,
    )
    freeze_backbones: Optional[str] = field(
        default=None
    )
    lr_scale: Optional[str] = field(
        default=None,
        metadata={'help': "available: [None, 'vision_model: 0.1, mlp: 1.0, llm: 1.0']"}
    )
    use_fp8: bool = field(
        default=False,
        metadata={'help': 'Set to True to use fp8.'}
    )
    mlp_connector_layers: int = field(
        default=2,
        metadata={'help': 'Set the number of MLP connector layers. Default is 2.'}
    )
    chat_template_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to chat template file.'}
    )
    processor_config_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to processor config file.'}
    )
    preprocessor_config_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to preprocessor config file.'}
    )
    block_size: int = field(
        default=4,
        metadata={'help': 'block size of mask token.'},
    )
    causal_attn: bool = field(
        default=False,
        metadata={'help': 'causal attention or not. default to False.'},
    )
    attn_implementation: Optional[str] = field(
        default='magi',
        metadata={'help': 'attention implementation: magi, flash_attention_2, sdpa, eager.'},
    )
    expected_mask_repeat_times: int = field(
        default=3,
        metadata={'help': 'expected mask repeat time.'},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    neftune_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'The noise_alpha value for NEFTune. Default is None.'},
    )
    n_frames: Optional[int] = field(
        default=16,
        metadata={'help': 'The number of frames for the video. Default is 16.'},
    )
    sequence_parallel_degree: int = field(
        default=1,
        metadata={'help': 'Sequence parallelism degree. Default is 1.'}
    )
    ring_sequence_parallel_degree: int = field(
        default=1,
        metadata={'help': 'Sequence parallelism ring degree. Default is 1.'}
    )
    sample_length_div: int = field(
        default=1,
        metadata={'help': 'Specify the division of sample length. Default is 1.'}
    )
    use_onelogger: bool = field(   
        default=False,
        metadata={'help': 'Set to True to use one logger.'}
    )
    use_online_packing: bool = field(
        default=True,
        metadata={'help': 'Set to True to use online packing.'}
    )
    video_total_pixels: int = field(
        default=16384 * 28 * 28 * 0.8,
        metadata={'help': 'The total number of pixels for the video. Default is 32000 * 28 * 28 * 0.9.'}
    )
    max_frames: int = field(
        default=64,
        metadata={'help': 'The maximum number of frames for the video. Default is 16.'}
    )
    target_fps: int = field(
        default=2,
        metadata={'help': 'The target FPS for the video. Default is 2.'}
    )
    max_num_tokens_per_sample: int = field(
        default=32768,
        metadata={'help': 'Maximum tokens allowed in one raw sample; longer samples are skipped.'}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={'help': 'Hard limit on tokens in a packed batch; flush if adding a sample would exceed it.'}
    )
    packing_buffer_size: int = field(
        default=32,
        metadata={'help': 'The buffer size for the packing. Default is 32.'}
    )
    sample_log_interval: int = field(
        default=100,
        metadata={'help': 'Log image count every N steps. Default is 100.'}
    )
    auto_thinking_handler: bool = field(
        default=False,
        metadata={'help': 'Set to True to auto handle thinking mode format. Default is False.'}
    )
