# !pip install isr --no-deps
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from glob import glob
import os

from ISR.models import RDN, RRDN
from config import inference_results

def upscale(model, patch_size=50):
    '''
    if image size is large then use patch size of your liking (recomended value = 50)

    '''
    imgs_list = glob(inference_results+'*')

    for i in range(0,len(imgs_list)):
        im_name = imgs_list[i]
        img = Image.open(im_name)
        if patch_size != None:
            pred_img = model.predict(np.array(img), by_patch_of_size=patch_size)
        else:
            pred_img = model.predict(np.array(img))

        #pred_img = model_noise.predict(pred_img, by_patch_of_size=patch_size)
        Image.fromarray(pred_img).save(inference_results+'upscaled_'+im_name.split('/')[-1],"PNG")



def main():
    
    #model = RDN(weights='noise-cancel')
    # model = RRDN(weights='gans')
    # model = RDN(weights='psnr-small')
    model = RDN(weights='psnr-large')
    # model = RRDN(weights='gans')
    
    upscale(model)

if __name__=="__main__":
    main()