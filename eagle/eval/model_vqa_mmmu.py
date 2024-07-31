import torch
import os
import random

import numpy as np
from tqdm import tqdm

# for debug
import sys
sys.path.append(os.getcwd())

from datasets import load_dataset, concatenate_datasets
from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import get_model_name_from_path
from eagle.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from argparse import ArgumentParser
from eval_utils.mmmu.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, process_multiple_sample, CAT_SHORT2LONG
from eval_utils.mmmu.model_utils import llava_image_processor
from eval_utils.mmmu.eval_utils import parse_multi_choice_response, parse_open_response

from eagle.utils import disable_torch_init
from eagle.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image

def call_eagle_engine_df(args, sample, model, tokenizer=None, processor=None):
    from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from eagle.conversation import conv_templates, SeparatorStyle

    def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image = sample['image']
    image_sizes = sample['image_size']

    if image is not None:
        #print('len(images):', len(image))
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_image_tensors(image, image_processor, model_config):
    # image = Image.open(image).convert('RGB')

    image_size = image.size
    image_tensor = process_images([image], image_processor, model_config)[0]

    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, image_size

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='playground/data/eval/mmmu/debug/output.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="eval_utils/mmmu/config/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="./playground/data/eval/MMMU") # hf dataset path.
    parser.add_argument('--model_path', type=str, default="checkpoints/finetune-llava-7b-336-full/final") # TODO: modify with a huggingface repo id
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print('eagle_initializing...')
    processor = None
    call_model_engine = call_eagle_engine_df
    vis_process_func = llava_image_processor

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values(): # 30 sub-categories in total
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    # load model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, vis_processors, _ = load_pretrained_model(args.model_path, None,
                                                                model_name)

    samples = []
    for sample in dataset:
        # sample = process_single_sample(sample)
        sample = process_multiple_sample(sample, add_visual_prompt=True)
        sample = construct_prompt(sample, args.config)

        if sample['image']:
            image, image_size = prepare_image_tensors(sample['image'].convert('RGB'),
                                                     image_processor=vis_processors,
                                                     model_config=model.config)
            sample['image'] = image
            sample['image_size'] = [image_size]
            samples.append(sample)

    # run ex
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)

    save_json(args.output_path, out_samples)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()
