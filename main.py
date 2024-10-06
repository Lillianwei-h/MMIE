import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from load_model import load_model
from prompts import get_prompt
from dataset import get_dataset
import json
from tqdm import tqdm
import argparse
import re
import os

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def find_score(text):
    match = re.search(r'### Score.*?(-?\d+(\.\d+)?)', text, re.DOTALL)
    if match:
        score = float(match.group(1))
        return round(score)
    else:
        return None

def average(scores):
    if len(scores) == 0:
        return None
    a = np.array(scores)
    mean = np.mean(a)
    return round(mean,3)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.

parser = argparse.ArgumentParser(description="MMIE Evaluation")

parser.add_argument('--model_path', type=str, default='MMIE-Eval', help='Path to the eval model directory')
parser.add_argument('--input_dir', type=str, default='./eval_inputs', help='Directory containing the input data')
parser.add_argument('--input_name', type=str, default='data.json', help='Name of the input data file')
parser.add_argument('--output_dir', type=str, default='./eval_outputs', help='Directory to save the output results')
parser.add_argument('--output_name', type=str, default='eval_result.json', help='Name of the output result file')
parser.add_argument('--save_step', type=int, default=10, help='Interval steps to save temp outputs')

args = parser.parse_args()

model_path = args.model_path
save_step = args.save_step
input_dir = args.input_dir
input_name = args.input_name
output_dir = args.output_dir
output_name = args.output_name
os.makedirs(os.path.join(output_dir, 'temp'), exist_ok=True)

output_path = os.path.join(output_dir, output_name)
temp_output_path = os.path.join(output_dir, 'temp', output_name)

model:AutoModel = load_model(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
generation_config = dict(max_new_tokens=1024, do_sample=False)

# If you have an interrupted evaluation,
# you can set 'temp_output_path=temp_output_path' to recover your last evaluation
processed_data = get_dataset(input_dir, input_name, temp_output_path=None)   
turn_step = 0
for d in tqdm(processed_data):
    if 'gpt_feedback' in d:
        continue
    question = d['question']
    answer = d['answer']
    images = d['images']
    system_prompt = get_prompt()
    # use ground truth as reference in multi-step reasoning
    if "MSR" in d['id']:
        gt_answer = d['gt_answer']
        prompt = system_prompt.format(question=question,answer=answer,gt_answer=gt_answer)
    else:
        prompt = system_prompt.format(question=question,answer=answer)
        
    pixel_values_list = []
    num_patches_list = []
    for image in images:
        pixel_value = load_image(image, max_num=12).to(torch.bfloat16).cuda()
        pixel_values_list.append(pixel_value)
        num_patches_list.append(pixel_value.size(0))
    pixel_values = torch.cat(pixel_values_list, dim=0)

    response, history = model.chat(tokenizer, pixel_values, prompt, generation_config,
                            num_patches_list=num_patches_list,
                            history=None, return_history=True)

    d["gpt_feedback"] = response
    turn_step += 1
    if turn_step == save_step:
        with open(temp_output_path,'w') as f:
            json.dump(processed_data, f, indent=4)
        turn_step = 0
        
with open(output_path,'w') as f:
    json.dump(processed_data, f, indent=4)

gpt_scores = []
sa_scores = []
pbl_scores = []
msr_scores = []
for d in processed_data:
    gpt_feedback = d['gpt_feedback']
    gpt_score = find_score(gpt_feedback)
    if isinstance(gpt_score,int) and gpt_score<=5 and gpt_score>=0:
        gpt_scores.append(gpt_score)
        if "SA" in d['id']:
            sa_scores.append(gpt_score)
        elif "PBL" in d['id']:
            pbl_scores.append(gpt_score)
        elif "MSR" in d['id']:
            msr_scores.append(gpt_score)
        else:
            print("Unrecogonized data.")

avg = average(gpt_scores)
sa_avg = average(sa_scores)
pbl_avg = average(pbl_scores)
msr_avg = average(msr_scores)
print("\n----------------------------------------------------------------------------------------")
print(f"File: {input_name}")
print(f"MMIE Eval Results\nSample size: {len(gpt_scores)}\nAVG: {avg}\nSA: {sa_avg}\nPBL: {pbl_avg}\nMSR: {msr_avg}")
print("----------------------------------------------------------------------------------------\n")
