from PIL import Image
import numpy as np
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import json
from keras.datasets import cifar10
from unet_strip import U_Net_In, U_Net_Out,AutoEncoder

import glob

def generate_data(minibatch):
    batch=min(len(glob.glob('Data/pre_valid_P/*.json')),minibatch)

    index=np.random.choice([i for i in range(len(glob.glob('Data/pre_valid_P/*.json')))],batch,replace=False)
    index=index.astype(np.int)
    filenames=[glob.glob('Data/pre_valid_P/*.json')[i] for i in index]

    training_data=[]
    for filename in filenames:
        with open(filename) as json_file:
            data=json.load(json_file)
            training_data.append(data)
    return training_data

def main():
    # for filename in glob.glob('Data/pre_valid_P/*.json'):
    #     with open(filename) as json_file:
    #         data=json.load(json_file)
    #
    #     patches=np.asarray(data['patches'])
    #     patchsize=data['patch_size']
    #     image_size=data['size']
    #
    #     image=np.zeros((patches.shape[0]*patchsize,patches.shape[1]*patchsize,3))
    #     for i in range(patches.shape[0]):
    #         for j in range(patches.shape[1]):
    #             image[patchsize*i:patchsize*(i+1),patchsize*j:patchsize*(j+1)]=patches[i][j]
    #
    #     image=image[:image_size[1],:image_size[0]].astype('uint8')
    #     im=Image.fromarray(image)
    #     im.save('temp/'+data['name'])
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    encoder=U_Net_In()
    decoder=U_Net_Out()
    encoder.load_weights('./saved_models/encoder.h5')
    decoder.load_weights('./saved_models/decoder.h5')
    for i in range(100):
        pass


if __name__ == "__main__":
    main()
