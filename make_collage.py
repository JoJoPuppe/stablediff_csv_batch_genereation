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

parser = argparse.ArgumentParser()
parser.add_argument('csvpath', help='path to csv file')
parser.add_argument('iter', help='number of iterations per prompt')
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


def read_csv(csv_path):
    prompt_obj_list = []
    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            prompt_obj_list.append(PromptObject(*row))

    return prompt_obj_list[1:3]

def collage_height(obj_list: List[PromptObject]):
    added_height = 0
    for obj in obj_list:
        added_height += int(obj.height)

    return added_height


def generate_images(url, promp_obj_list: List[PromptObject], num_iterations=5):
    image_list = []
    for prompt in promp_obj_list:
        image_row = []
        for iter in range(num_iterations):

            if iter > 0:
                use_seed = int(prompt.seed) + SEED_DIFFS[iter - 1]
            else:
                use_seed = int(prompt.seed)
            print(use_seed)
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
            image_row.append(generate_image(url, payload))
        image_list.append(image_row)
    return image_list


def generate_image(url, payload):
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        return image


def generate_art_sheet(images, height):
    width = len(images[0]) * 512
    print(f"collage_width = {width}, {height}")
    collage = Image.new("RGBA", (width, height), color=(255,255,255,255))
    for img_row in range(len(images)):
        for img_idx in range(len(images[img_row])):
            print(images[img_row][img_idx].size)
            if img_row == 0:
                y = 0
            else:
                y = img_row * images[img_row - 1][img_idx].size[1]
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
    promp_obj_list = read_csv(args.csvpath)
    images = generate_images(url, promp_obj_list, num_iterations=int(args.iter))
    height = collage_height(promp_obj_list)
    generate_art_sheet(images, height)
