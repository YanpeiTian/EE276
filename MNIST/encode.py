from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
import data
import dense_models, conv_models
import numpy as np

import matplotlib.pyplot as plt

def main():
    # _,x_test=data.get_data()
    # x_test = x_test.reshape((len(x_test), 28,28,1))

    encoder=load_model('models/encoder.h5')
    encoder.summary()

    # encoded_imgs = encoder.predict(x_test)

    # np.save('data/encoded_imgs',encoded_imgs)


if __name__ == "__main__":
    main()
