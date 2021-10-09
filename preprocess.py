from PIL import ImageFile, Image
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, sys
from tqdm import tqdm
from config import *


if not os.path.isdir(resized_path):
    os.makedirs(resized_path)

dirs = os.listdir(orig_path)

for item in tqdm(dirs):
    if os.path.isfile(orig_path+item):
        im = Image.open(orig_path+item).convert('RGB')
        f, e = os.path.splitext(item)
        imResize = im.resize((width,height), Image.ANTIALIAS)
        imResize.save(resized_path+ f +".jpg", 'JPEG', quality=95)
