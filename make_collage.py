


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
from datetime import datetime

from timeit import default_timer as timer
from datetime import timedelta


parser = argparse.ArgumentParser()
parser.add_argument('csvpath', help='path to csv file')
parser.add_argument('iter', help='number of iterations per prompt')
parser.add_argument('stylecnt', help='number of styles')
parser.add_argument('name', help='name of person where the model gets trained on')
args = parser.parse_args()

# prompt,seed,width,height,sampler,cfgs,steps,filename,negative_prompt

url = "http://127.0.0.1:7860"

SEED_DIFFS = [2, -2, 81, 4, -9, 4816, -4771, 3123, 10233, -10222]
BASEMODEL_STRING = "marcusloeper"
BASE_OUT_PATH = pathlib.Path("./out")
MAX_WIDTH = 800

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
    model_hash: str
    restore_faces: str
    denoising_strength: str


def read_csv(csv_path, style_cnt):
    prompt_obj_list = []
    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            prompt_obj_list.append(PromptObject(*row))

    return prompt_obj_list[1:style_cnt + 1]

def get_current_model():
    response = requests.get(url=f'{url}/sdapi/v1/options')
    json_response = response.json()
    return json_response['sd_model_checkpoint']

def set_checkpoint_model(url, model_string):
    option_payload = {
        "sd_model_checkpoint": model_string,
    }
    response = requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)
    return response


def get_all_model_names(url):
    response = requests.get(url=f'{url}/sdapi/v1/sd-models')
    # data = response.json()
    return response.json()

def get_model_title(all_model_names, hash):
    new_model = None
    for model in all_model_names:
        if model['hash'] == hash:
            print(f"found hash: {hash}")
            model_name = model["model_name"].replace(BASEMODEL_STRING, name)
            for m in all_model_names:
                if model_name == m["model_name"]:
                    new_model = m["title"]
                    print(f"new model: {new_model}")
                    return new_model
    return new_model

def get_model_string(url, name, hash):
    all_model_names = get_all_model_names(url)
    all_titles = [i['title'] for i in all_model_names]
    new_model = get_model_title(all_model_names, hash)
    if new_model is not None and new_model in all_titles:
        return new_model
    elif new_model is not None:
        for t in all_titles:
            if t.find(name) != -1:
                print(f'model not found using: {t}')
                return t
    else:
        raise Exception("no model with {name} found. please train dreambooth!")

def generate_images(url, promp_obj_list: List[PromptObject], folder, num_iterations=5, name=None):
    start_all = timer()
    currt_cnt = 1
    image_list = []
    sum_img = num_iterations * len(promp_obj_list)
    current_model = get_current_model()
    for idx, prompt in enumerate(promp_obj_list):
        image_row = []
        for iter in range(num_iterations):
            if iter > 0:
                use_seed = int(prompt.seed) + SEED_DIFFS[iter - 1]
            else:
                use_seed = int(prompt.seed)

            if name is not None:
                new_model = get_model_string(url, name, prompt.model_hash)
                if new_model != current_model:
                    set_checkpoint_model(url, new_model)
                prompt.prompt = prompt.prompt.replace(BASEMODEL_STRING, name)
            payload = {
                "prompt": prompt.prompt,
                "steps": int(prompt.steps),
                "seed": use_seed,
                "cfg_scale": float(prompt.cfgs),
                "negative_prompt": prompt.negative_prompt,
                "width": int(prompt.width),
                "height": int(prompt.height),
                "sampler_index": prompt.sampler,
                "restore_faces": prompt.restore_faces
            }
            if prompt.denoising_strength != '':
                payload['denoising_strength'] = float(prompt.denoising_strength)
            start = timer()

            image_row.append(generate_image(url, payload, folder, idx, iter))

            end = timer()
            print(f'{currt_cnt}/{sum_img} - {prompt.prompt[:20]}... - {timedelta(seconds=end-start)} - ges: {timedelta(seconds=end - start_all)}')
            currt_cnt += 1
        image_list.append(image_row)
    return image_list


def generate_image(url, payload, folder, idx, iter):
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    print(r)
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        save_image(url, image, i, folder, idx, iter)
        return image


def resize_all_images(images):
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

def split_list(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]

def generate_art_sheet(images, folder):
    row_size = 4
    all_collages = list(split_list(images, row_size))
    for id, collage_image in enumerate(all_collages):
        height = sum(i[0].size[1] for i in collage_image)
        width = sum(i.size[0] for i in collage_image[0])
        print(f"collage-{id+1}_width = {width}, {height}")
        collage = Image.new("RGBA", (width, height), color=(255,255,255,255))
        y = 0
        for img_row in range(len(collage_image)):
            if img_row == 0:
                y = 0
            else:
                y += collage_image[img_row - 1][0].size[1]
            for img_idx in range(len(collage_image[img_row])):
                x = img_idx * collage_image[img_row][img_idx].size[0]
                collage.paste(collage_image[img_row][img_idx], (x,y))

        collage_out = pathlib.Path.joinpath(folder, f"collage-{id+1}.png")
        collage.save(collage_out)


def save_image(url, image, i, folder, idx, iter):

    row_folder = pathlib.Path.joinpath(folder, str(idx))
    row_folder.mkdir(parents=True, exist_ok=True)
    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response.json().get("info"))
    out_file_path = pathlib.Path.joinpath(row_folder, f"{iter}-out.png")
    image.save(out_file_path, pnginfo=pnginfo)

if __name__ == "__main__":
    promp_obj_list = read_csv(args.csvpath, int(args.stylecnt))
    name = None
    if args.name:
        name = args.name
    else:
        raise Exception("please choose a model name.")

    pathlib.Path('/tmp/sub1/sub2').mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    folder_name = pathlib.Path.joinpath(BASE_OUT_PATH, f"{name}-{now.strftime('%Y%m%d-%H%M%S')}")
    folder_name.mkdir(parents=True, exist_ok=True)

    images = generate_images(url, promp_obj_list, folder_name, num_iterations=int(args.iter), name=name)
    images = resize_all_images(images)
    generate_art_sheet(images, folder_name)
