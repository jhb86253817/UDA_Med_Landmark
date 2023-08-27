# -*- coding: utf-8 -*-

import os, cv2 
import pathlib
import re
import numpy as np

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

img_path = "CXR_png/"

data_root = pathlib.Path(img_path)
all_files = list(data_root.glob('*.png'))
all_files = [str(path) for path in all_files]
all_files.sort(key = natural_key)

save_img = "Images/"
if not os.path.exists(save_img):
    os.mkdir(save_img)

i = 1

for file in all_files:
    print('File', i, 'of', len(all_files))

    img = cv2.imread(file, 0)

    gray = 255*(img > 1) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)

    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    cropimg = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image

    shape = cropimg.shape

    if shape[0] < shape[1]:
        pad = (shape[1] - shape[0])    
        
        if pad % 2 == 1:
            pad = pad // 2
            pad_y = [pad, pad+1]
        else:
            pad = pad // 2
            pad_y = [pad, pad]
            
        pad_x = [0, 0]
    elif shape[1] < shape[0]:
        pad = (shape[0] - shape[1]) 
        
        if pad % 2 == 1:
            pad = pad // 2
            pad_x = [pad, pad+1]
        else:
            pad = pad // 2
            pad_x = [pad, pad]
            
        pad_y = [0, 0]

    img = np.pad(cropimg, pad_width = [pad_y, pad_x])    

    if img.shape[0] != img.shape[1]:
        print('Error padding image')
        break

    img_ = cv2.resize(img, [1024, 1024])
    
    cv2.imwrite(file.replace(img_path, save_img), img_)

    i = i+1
