# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

def preprocess(folderpath, flist):
    for f in flist:
        p = os.path.join('All247images', f)
        
        w, h = 2048, 2048 

        with open(p, 'rb') as path: 
            dtype = np.dtype('>u2')
            img = np.fromfile(path, dtype=dtype).reshape((h,w)) 

        img = 1 - img.astype('float')  / 4096
        img = cv2.resize(img, (1024,1024))
        img = img*255
       
        p = os.path.join(folderpath, f.replace('.IMG','.png'))
        cv2.imwrite(p, img.astype('uint8'))

image_names = os.listdir('All247images')
folder_out = "Images"
if not os.path.exists(folder_out):
    os.mkdir(folder_out)
preprocess(folder_out, image_names)

anno_names = os.listdir('annos')
image_names = os.listdir(folder_out)
for image_name in image_names:
    anno_name = image_name[:-3]+'npy'
    if anno_name not in set(anno_names):
        os.system('rm {}/{}'.format(folder_out, image_name))
        print('removed extra image: {}'.format(image_name))

print("images preprocessed")
