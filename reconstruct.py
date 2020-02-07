from PIL import Image
import numpy as np
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import json

import glob

for filename in glob.glob('Data/pre_valid_P/*.json'):
    with open(filename) as json_file:
        data=json.load(json_file)

    patches=np.asarray(data['patches'])
    patchsize=data['patch_size']
    image_size=data['size']

    image=np.zeros((patches.shape[0]*patchsize,patches.shape[1]*patchsize,3))
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            image[patchsize*i:patchsize*(i+1),patchsize*j:patchsize*(j+1)]=patches[i][j]

    image=image[:image_size[1],:image_size[0]].astype('uint8')
    im=Image.fromarray(image)
    im.save('temp/'+data['name'])
