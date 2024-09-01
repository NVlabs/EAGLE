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
# This file is modified from https://huggingface.co/spaces/shi-labs/CuMo-7b-zero/blob/main/app.py

import gradio as gr
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

from PIL import Image
import argparse

from transformers import TextIteratorStreamer
from threading import Thread

# os.environ['GRADIO_TEMP_DIR'] = './gradio_tmp'
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

argparser = argparse.ArgumentParser()
argparser.add_argument("--server_name", default="0.0.0.0", type=str)
argparser.add_argument("--port", default="6324", type=str)
argparser.add_argument("--model-path", default="NVEagle/Eagle-X5-13B-Chat", type=str)
argparser.add_argument("--model-base", type=str, default=None)
argparser.add_argument("--num-gpus", type=int, default=1)
argparser.add_argument("--conv-mode", type=str, default="vicuna_v1",)
argparser.add_argument("--temperature", type=float, default=0.2)
argparser.add_argument("--max-new-tokens", type=int, default=512)
argparser.add_argument("--num_frames", type=int, default=16)
argparser.add_argument("--load-8bit", action="store_true")
argparser.add_argument("--load-4bit", action="store_true")
argparser.add_argument("--debug", action="store_true")

args = argparser.parse_args()
model_path = args.model_path
conv_mode = args.conv_mode
filt_invalid="cut"
model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
our_chatbot = None

def upvote_last_response(state):
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state):
    return ("",) + (disable_btn,) * 3


def flag_last_response(state):
    return ("",) + (disable_btn,) * 3

def clear_history():
    state =conv_templates[conv_mode].copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def add_text(state, imagebox, textbox, image_process_mode):
    if state is None:
        state = conv_templates[conv_mode].copy()

    if imagebox is not None:
        textbox = DEFAULT_IMAGE_TOKEN + '\n' + textbox
        image = Image.open(imagebox).convert('RGB')

    if imagebox is not None:
        textbox = (textbox, image, image_process_mode)

    state.append_message(state.roles[0], textbox)
    state.append_message(state.roles[1], None)

    yield (state, state.to_gradio_chatbot(), "", None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)

def delete_text(state, image_process_mode):
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    yield (state, state.to_gradio_chatbot(), "", None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)

def regenerate(state, image_process_mode):
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

# @spaces.GPU
def generate(state, imagebox, textbox, image_process_mode, temperature, top_p, max_output_tokens):
    prompt = state.get_prompt()
    images = state.get_images(return_pil=True)
    #prompt, image_args = process_image(prompt, images)

    ori_prompt = prompt
    num_image_tokens = 0

    if images is not None and len(images) > 0:
        if len(images) > 0:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError("Number of images does not match number of <image> tokens in prompt")
            
            #images = [load_image_from_base64(image) for image in images]
            image_sizes = [image.size for image in images]
            images = process_images(images, image_processor, model.config)

            if type(images) is list:
                images = [image.to(model.device, dtype=torch.float16) for image in images]
            else:
                images = images.to(model.device, dtype=torch.float16)
        else:
            images = None
            image_sizes = None
        image_args = {"images": images, "image_sizes": image_sizes}
    else:
        images = None
        image_args = {}

    max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
    max_new_tokens = 512
    do_sample = True if temperature > 0.001 else False
    stop_str = state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

    max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

    if max_new_tokens < 1:
        # yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
        return

    thread = Thread(target=model.generate, kwargs=dict(
        inputs=input_ids,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        **image_args
    ))
    thread.start()
    generated_text = ''
    for new_text in streamer:
        generated_text += new_text
        if generated_text.endswith(stop_str):
            generated_text = generated_text[:-len(stop_str)]
        state.messages[-1][-1] = generated_text
        yield (state, state.to_gradio_chatbot(), "", None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
    
    yield (state, state.to_gradio_chatbot(), "", None) + (enable_btn,) * 5
    
    torch.cuda.empty_cache()

txt = gr.Textbox(
    scale=4,
    show_label=False,
    placeholder="Enter text and press enter.",
    container=False,
)


title_markdown = ("""
# Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders
[[Code](https://github.com/NVlabs/EAGLE)] [[Model](https://huggingface.co/NVEagle)] | 📚 [[Arxiv](https://arxiv.org/pdf/2408.15998)]]
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the. Please contact us if you find any potential violation.
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
with gr.Blocks(title="Eagle", theme=gr.themes.Default(), css=block_css) as demo:
    state = gr.State()

    gr.Markdown(title_markdown)

    with gr.Row():
        with gr.Column(scale=3):
            imagebox = gr.Image(label="Input Image", type="filepath")
            image_process_mode = gr.Radio(
                ["Crop", "Resize", "Pad", "Default"],
                value="Default",
                label="Preprocess for non-square image", visible=False)

    
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            gr.Examples(examples=[
                [f"{cur_dir}/assets/health-insurance.png", "Under which circumstances do I need to be enrolled in mandatory health insurance if I am an international student?"],
                [f"{cur_dir}/assets/leasing-apartment.png", "I don't have any 3rd party renter's insurance now. Do I need to get one for myself?"],
                [f"{cur_dir}/assets/nvidia.jpeg", "Who is the person in the middle?"],
                [f"{cur_dir}/assets/animal-compare.png", "Are these two pictures showing the same kind of animal?"],
                [f"{cur_dir}/assets/georgia-tech.jpeg", "Where is this photo taken?"]
            ], inputs=[imagebox, textbox], cache_examples=False)

            with gr.Accordion("Parameters", open=False) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Eagle Chatbot",
                height=650,
                layout="panel",
            )
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(value="Send", variant="primary")
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)
    url_params = gr.JSON(visible=False)

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state],
        [textbox, upvote_btn, downvote_btn, flag_btn]
    )
    downvote_btn.click(
        downvote_last_response,
        [state],
        [textbox, upvote_btn, downvote_btn, flag_btn]
    )
    flag_btn.click(
        flag_last_response,
        [state],
        [textbox, upvote_btn, downvote_btn, flag_btn]
    )

    clear_btn.click(
        clear_history,
        None,
        [state, chatbot, textbox, imagebox] + btn_list,
        queue=False
    )

    regenerate_btn.click(
        delete_text,
        [state, image_process_mode],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        generate,
        [state, imagebox, textbox, image_process_mode, temperature, top_p, max_output_tokens],
        [state, chatbot, textbox, imagebox] + btn_list,
    )
    textbox.submit(
        add_text,
        [state, imagebox, textbox, image_process_mode],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        generate,
        [state, imagebox, textbox, image_process_mode, temperature, top_p, max_output_tokens],
        [state, chatbot, textbox, imagebox] + btn_list,
    )

    submit_btn.click(
        add_text,
        [state, imagebox, textbox, image_process_mode],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        generate,
        [state, imagebox, textbox, image_process_mode, temperature, top_p, max_output_tokens],
        [state, chatbot, textbox, imagebox] + btn_list,
    )

demo.queue(
    status_update_rate=10,
    api_open=False
).launch(share=True)
demo.queue()

# if __name__ == "__main__":

#     # import pdb;pdb.set_trace()
#     try:
#         demo.launch(server_name=args.server_name, server_port=int(args.port), share=True)
#     except Exception as e:
#         args.port=int(args.port)+1
#         print(f"Port {args.port} is occupied, try port {args.port}")
#         demo.launch(server_name=args.server_name, server_port=int(args.port), share=True)
