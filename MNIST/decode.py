from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import save_img
import data
import dense_models, conv_models
import numpy as np

import matplotlib.pyplot as plt

def main():
    _,x_test=data.get_data()
    # x_test = x_test.reshape((len(x_test), 28,28,1))

    decoder=load_model('models/decoder.h5')

    encoded_imgs=np.load('data/encoded_imgs.npy')

    reconstructions = decoder.predict(encoded_imgs)
    reconstructions = reconstructions.reshape((len(x_test),28,28))
    reconstructions = 255*reconstructions

    for i in range(len(reconstructions)):

        temp=np.expand_dims(reconstructions[i],axis=2)

        temp=np.repeat(temp.astype(np.uint8), 3, 2)
        save_img('data/'+str(i)+'.png',temp)

if __name__ == "__main__":
    main()
