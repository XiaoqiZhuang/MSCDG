import os
import sys
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, DiffusionPipeline
import time
import argparse
from PIL import Image
import numpy as np
from torchvision import transforms
import time

from pipeline_MCOW_BETTER import MCOWPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = MCOWPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to(device)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)

s1 = 'cat'
s2 = 'dog'
seed_size = 200

img1 = Image.open(f'data/images/a {s1} on the grass.jpg')
mask1 = Image.open(f'data/masks/{s1}.jpg')

img2 = Image.open(f'data/images/a {s2} on the grass.jpg')
mask2 = Image.open(f'data/masks/{s2}.jpg')

p1 = f'a {s1} are sitting on the grass'
p2 = f'a {s2} are sitting on the grass'
p = f' a {s1} and a {s2} are sitting on the grass'

for seed in range(10):
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipeline(prompt=p, num_inference_steps=10, generator=generator,
                    subprompts=[p1, p2], images=[img1, img2], mask_images=[mask1, mask2], image_sizes=[seed_size, seed_size], xys=[[0,200], [255, 200]],)[0]

    # image = pipeline(prompt=p1, num_inference_steps=10, generator=generator,
    #                  subprompts=[p1], images=[img1], mask_images=[mask1], image_sizes=[seed_size], xys=[[128,200]],)[0]

    image.save(f"results/{p}_{seed}.jpg")

