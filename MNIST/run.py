from keras.layers import Input, Dense
from keras.models import Model
import data
import dense_models, conv_models
import numpy as np

import matplotlib.pyplot as plt

# Models summary:
# dense_models:
#   1. single_layer
#   2. two_layer
#   3. deep_model (3 layer)
# conv_models:
#   1. conv_model (3 layer)
#   2. mixed_model (3+2 layer)

def train(model,x_train,x_test):
    model.fit(x_train, x_train,epochs=50,batch_size=256,shuffle=True,validation_data=(x_test, x_test))

def test(encoder,decoder,x_test):

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)


    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    print("The compression rate is: %f",x_test.size/encoded_imgs.size)
    print("The MSE loss on test set is: %f",np.mean(np.multiply(x_test-decoded_imgs,x_test-decoded_imgs)))

def main():

    x_train,x_test=data.get_data()

    # model,encoder,decoder=dense_models.single_layer()
    # 50,992 parameters
    # compression ratio=24.5 ; mse loss=0.023465356

    # model,encoder,decoder=dense_models.two_layer()
    # 793,792 parameters
    # compression ratio= 98; mse loss=0.016756587

    # model,encoder,decoder=dense_models.deep_model()
    # 222,384 parameters
    # compression ratio= 24.5; mse loss=0.008605147

    model,encoder,decoder=conv_models.conv_model()
    x_train = x_train.reshape((len(x_train), 28,28,1))
    x_test = x_test.reshape((len(x_test), 28,28,1))
    # 4,385 parameters
    # compression ratio= 6.125; mse loss=0.009870365

    # model,encoder,decoder=conv_models.mixed_model()
    # x_train = x_train.reshape((len(x_train), 28,28,1))
    # x_test = x_test.reshape((len(x_test), 28,28,1))
    # 13.289 parameters
    # compression ratio= 98; mse loss=0.0266319

    print("Training set shape: %f",x_train.shape)
    print("Testing set shape: %f",x_test.shape)

    train(model, x_train,x_test)

    test(encoder,decoder,x_test)


if __name__ == "__main__":
    main()
