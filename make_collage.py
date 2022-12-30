import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import csv
import argparse
import pathlib
from dataclasses import dataclass
from typing import List

from timeit import default_timer as timer
from datetime import timedelta


parser = argparse.ArgumentParser()
parser.add_argument('csvpath', help='path to csv file')
parser.add_argument('iter', help='number of iterations per prompt')
parser.add_argument('stylecnt', help='number of styles')
args = parser.parse_args()

# prompt,seed,width,height,sampler,cfgs,steps,filename,negative_prompt

url = "http://127.0.0.1:7860"

SEED_DIFFS = [2, -2, 81, 4, -9, 4816, -4771, 3123, 10233, -10222]


@dataclass
class PromptObject:
    prompt: str
    seed: str
    width: str
    height: str
    sampler: str
    cfgs: str
    steps: str
    filename: str
    negative_prompt: str


def read_csv(csv_path, style_cnt):
    prompt_obj_list = []
    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            prompt_obj_list.append(PromptObject(*row))

    return prompt_obj_list[1:style_cnt + 1]

def generate_images(url, promp_obj_list: List[PromptObject], num_iterations=5):
    image_list = []
    sum_img = num_iterations * len(promp_obj_list)
    for idx, prompt in enumerate(promp_obj_list):
        image_row = []
        for iter in range(num_iterations):

            if iter > 0:
                use_seed = int(prompt.seed) + SEED_DIFFS[iter - 1]
            else:
                use_seed = int(prompt.seed)
            payload = {
                "prompt": prompt.prompt,
                "steps": int(prompt.steps),
                "seed": use_seed,
                "cfg_scale": float(prompt.cfgs),
                "negative_prompt": prompt.negative_prompt,
                "width": int(prompt.width),
                "height": int(prompt.height),
                "sampler_index": prompt.sampler
            }
            start = timer()

            image_row.append(generate_image(url, payload))

            end = timer()
            currt_cnt = (idx + 1) * (iter + 1)
            print(f'{currt_cnt}/{sum_img} - {prompt.prompt[:20]}... - {timedelta(seconds=end-start)}')
        image_list.append(image_row)
    return image_list


def generate_image(url, payload):
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        return image


def resize_all_images(images):
    MAX_WIDTH = 800
    resized_images = []
    scale_factor = None
    comb_width = sum(i.size[0] for i in images[0])
    if comb_width > MAX_WIDTH:
        scale_factor = round(MAX_WIDTH / comb_width, 1)

    if scale_factor is None:
        return images

    for row in images:
        new_row = []
        for img in row:
            n_width = round(img.size[0] * scale_factor)
            n_height = round(img.size[1] * scale_factor)
            new_img = img.resize((n_width, n_height), Image.LANCZOS)
            new_row.append(new_img)
        resized_images.append(new_row)
    return resized_images


def generate_art_sheet(images):
    height = sum(i[0].size[1] for i in images)
    width = sum(i.size[0] for i in images[0])
    print(f"collage_width = {width}, {height}")
    collage = Image.new("RGBA", (width, height), color=(255,255,255,255))
    for img_row in range(len(images)):
        y = 0
        if img_row == 0:
            y = 0
        else:
            y += images[img_row - 1][0].size[1]
        for img_idx in range(len(images[img_row])):
            x = img_idx * images[img_row][img_idx].size[0]
            collage.paste(images[img_row][img_idx], (x,y))

    collage.show()
    collage.save("collage.png")

def generate_single_art(payload):
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()

    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

        png_payload = {
            "image": "data:image/png;base64," + i
        }
        response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image.save('output.png', pnginfo=pnginfo)

if __name__ == "__main__":
    promp_obj_list = read_csv(args.csvpath, int(args.stylecnt))
    images = generate_images(url, promp_obj_list, num_iterations=int(args.iter))
    images = resize_all_images(images)
    generate_art_sheet(images)
