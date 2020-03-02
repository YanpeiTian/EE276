import os
import pdb
import keras
from keras.datasets import cifar10,mnist
from keras.preprocessing.image import ImageDataGenerator,save_img
import tensorflow as tf
# from unet import U_Net
from unet_strip import U_Net_In, U_Net_Out,AutoEncoder
import numpy as np

# from conv_models_cifar import conv_model, mixed_model, DQN_model

BATCH_SIZE = 64
EPOCH_NUM = 1
MAX_ITER = 1000

def main():
    # data loading
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_test /=255
    x_train /=255
    x_train=np.pad(x_train,((0,0),(2,2),(2,2)),'constant',constant_values=0)
    x_test=np.pad(x_test,((0,0),(2,2),(2,2)),'constant',constant_values=0)
    x_train =x_train.reshape((-1,32,32,1))
    x_test =x_test.reshape((-1,32,32,1))

    # model
    ae=AutoEncoder()

    # def ssim_loss(img1,img2):
    #     return tf.image.ssim(img1, img2, max_val=1)

    # autoencoder.compile(optimizer='adam', loss=ssim_loss)
    ae.compile(optimizer='adam', loss='mean_squared_error')
    ae.fit(x_train[:100], x_train[:100], epochs=EPOCH_NUM, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test))
    #
    ae.summary()
    # ae.encoder.save_weights(os.path.join(save_dir, 'encoder.h5'))
    # ae.decoder.save_weights(os.path.join(save_dir, 'decoder.h5'))

    ae.encoder.load_weights('./mnist/encoder.h5')
    ae.decoder.load_weights('./mnist/decoder.h5')
    encoded_imgs = ae.encoder(x_test[:100])
    decoded_imgs = ae.decoder(encoded_imgs)
    print(sum(tf.keras.backend.eval(decoded_imgs[0])))
    for i in range(100):
        save_img('data/'+str(i)+'.png',x_test[i])
        temp=tf.keras.backend.eval(decoded_imgs[i])
        save_img('data/'+str(i)+'re.png',temp)
    print("finish")

if __name__ == '__main__':
    main()
