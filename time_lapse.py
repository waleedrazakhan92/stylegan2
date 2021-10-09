import cv2
from os import listdir
from os.path import isfile, join
from google.colab import files
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import imageio
import os

from config import results_dir


result_path = results_dir
image_files = [f for f in listdir(result_path)]
image_files = [f for f in image_files if '.png' in f]

images = []

width = 256
width_images = 8
width_offset = 0 * width
end_width = np.max(2048, width_images * width) 

height = 256
height_images = 4
height_offset = 0 * height
end_height = np.max(1280, height_images * height)


for f in tqdm(image_files[:4]):
  name = "{}{}".format(result_path, f)
  print(name)
  img = Image.open(name)
  img = img.crop((0, 0, end_width+ width_offset, end_height + height_offset))
  img = img.resize((1920,1080))
  images.append(img)

try:
  os.mkdir("out/")
except:
  pass

video_output_path = 'out/'
video_name = 'timelapse_movie.mp4'
movie_name = video_output_path + video_name

with imageio.get_writer(movie_name, mode='I') as writer:
    for image in tqdm(list(images)):
        writer.append_data(np.array(image))
