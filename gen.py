# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:33:14 2021

@author: sas11
"""


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
from imutils import paths
import PIL.Image

"""load image from source
"""
imagePaths = list(paths.list_images("1"))

i = 1

for imagePath in imagePaths:
    # extract the class label from the filename
    image = load_img(imagePath,target_size=(48,48))
    #image = np.rot90(image,1)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    

	# construct the image generator for data augmentation then
	# initialize the total number of images generated thus far
    gen = ImageDataGenerator(
        featurewise_center=True,
       featurewise_std_normalization=True,
       brightness_range=[0.4,1.0],
       rotation_range=25,
       width_shift_range=0.2,
       height_shift_range=0.2,
       horizontal_flip=True,
       shear_range=0.1,
       zoom_range=0.1,
   		fill_mode="nearest")
    j = 0
    i+=1
    #print(image)
    # construct the actual Python generator
    if i == 220:
        break
    for batch in gen.flow(image, batch_size=1, save_to_dir="dataset/",
                      save_prefix=str(i)+str(j), save_format="jpg") :

        j+= 1
        """change value"""
        if j == 1:
            break