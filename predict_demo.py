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
# A simple reference example of using eagle model

import os
import torch
import numpy as np

from eagle import conversation as conversation_lib
from eagle.constants import DEFAULT_IMAGE_TOKEN

from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from eagle.conversation import conv_templates, SeparatorStyle
from eagle.model.builder import load_pretrained_model
from eagle.utils import disable_torch_init
from eagle.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images, KeywordsStoppingCriteria

from PIL import Image
import argparse

from transformers import TextIteratorStreamer
from threading import Thread

model_path = "NVEagle/Eagle-X5-13B-Chat"
conv_mode = "vicuna_v1"
image_path = "assets/georgia-tech.jpeg"
input_prompt = "Describe this image."

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
                                                                       None, 
                                                                       model_name, 
                                                                       False, 
                                                                       False)

if model.config.mm_use_im_start_end:
    input_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + input_prompt
else:
    input_prompt = DEFAULT_IMAGE_TOKEN + '\n' + input_prompt

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], input_prompt)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

image = Image.open(image_path).convert('RGB')
image_tensor = process_images([image], image_processor, model.config)[0]
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

input_ids = input_ids.to(device='cuda', non_blocking=True)
image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids.unsqueeze(0),
        images=image_tensor.unsqueeze(0),
        image_sizes=[image.size],
        do_sample=True,
        temperature=0.2,
        top_p=0.5,
        num_beams=1,
        max_new_tokens=256,
        use_cache=True)

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(f"Image:{image_path} \nPrompt:{input_prompt} \nOutput:{outputs}")