# stylegan2
**Note** 
This repository is just a refactored version of StyleGAN2 from NVlabs and idealo's Image Super-resolution models. Most of the code is taken from the original repositories https://github.com/NVlabs/stylegan2-ada and https://github.com/idealo/image-super-resolution. The code is changed a bit to make it user friendly for beginners.

**Overview**
This repository aims to provide a method to generate high resolution synthetic images by combining the powers of StyleGAN2-ada and Image Super-resolution. StyleGAN2 is trained on a desired set of images to generate more of the kind but even the stylegan2-ada takes a lot of memory and time to train. So what this repository proposes is that you can train the stylegan2 on slightely low resolution images, e.g 256,256, and then apply the superpixel model to convert the images to higher resolutions, thus saving time and memory. 

# Usage
Clone the repository and change the current directory to this repository:
```
cd stylegan2
```
Run the bash script 'run_init' to initialize the process of downloading the pretrained models and the original stylegan repositories.
```
./run_init
```
Place the dataset in a folder and set ***orig_path*** in the *config.py* file to the dataset folder and change the ***height*** and ***width*** variables according to your output resolution requirements. 
Then run the bash script ```run_preprocess``` to preprocess the images into desired resolution. You can set the prath of preprocessed images by setting the ****resized path** in the config.py file.
```
./run_preprocess
```
The next step is to generate *tfrecords* using the script '''run_generate_tf_record'''. This will start the process of generating tfrecords for the model to train on. You can set the path of tfrecords by setting the ***tf_record_path*** in the config.py file. 
'''
./run_generate_tf_record
'''
Next you have to select a pretrained checkpoint ```.pkl``` file. By default the code downloads three checkpoint files with 256, 512, 1024 image resolutions respectively. If your
