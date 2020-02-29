import os
import pdb
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from conv_models_cifar import conv_model, mixed_model, DQN_model

# data loading
BATCH_SIZE = 64 
EPOCH_NUM = 100
MAX_ITER = 1000
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# model
autoencoder, encoder, decoder = conv_model()

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=EPOCH_NUM, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test))

encoder.save(os.path.join(save_dir, 'encoder.h5'))
decoder.save(os.path.join(save_dir, 'decoder.h5'))

if __name__ == '__main__':
    main()
