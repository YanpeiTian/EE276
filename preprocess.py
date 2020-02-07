from PIL import Image
import numpy as np
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import json

import glob

PATCHSIZE = 128

for filename in glob.glob('Data/valid_P/*.png'):
    # Open an image
    im=Image.open(filename)

    # build the dictionary
    dict={}
    # filename
    temp=filename[filename.index('/')+1::]
    dict['name']=temp[temp.index('/')+1:]
    # image size
    dict['size']=im.size
    # patch size
    dict['patch_size']=PATCHSIZE

    # preprocess image into patches
    data=np.asarray(im)

    dim1=int(np.ceil(data.shape[0]/PATCHSIZE))
    dim2=int(np.ceil(data.shape[1]/PATCHSIZE))

    patches=np.zeros((dim1,dim2,PATCHSIZE,PATCHSIZE,3))
    data=np.pad(data,((0,dim1*PATCHSIZE-im.size[1]),(0,dim2*PATCHSIZE-im.size[0]),(0,0)),'constant')

    for i in range(dim1):
        for j in range(dim2):
            patches[i][j]=data[PATCHSIZE*i:PATCHSIZE*(i+1),PATCHSIZE*j:PATCHSIZE*(j+1)]

    dict['patches']=patches.astype('uint8')

    # save file
    dict['patches']=dict['patches'].tolist()
    with open('Data/pre_valid_P/'+dict['name'][:-4]+'.json', 'w') as fp:
        json.dump(dict, fp)
