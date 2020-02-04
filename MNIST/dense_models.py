from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers,optimizers

import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers

def single_layer():

    # this is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(784,))

    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))

    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]

    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.summary()

    return autoencoder,encoder,decoder

def two_layer():

    # Encoder
    # x = tensorflow.keras.layers.Input(shape=(784), name="encoder_input")
    #
    # encoder_dense_layer1 = tensorflow.keras.layers.Dense(units=500, name="encoder_dense_1")(x)
    # encoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_dense_layer1)
    #
    # encoder_dense_layer2 = tensorflow.keras.layers.Dense(units=8, name="encoder_dense_2")(encoder_activ_layer1)
    # encoder_output = tensorflow.keras.layers.LeakyReLU(name="encoder_output")(encoder_dense_layer2)
    # encoder = tensorflow.keras.models.Model(x, encoder_output, name="encoder_model")

    input_img = Input(shape=(784,))
    encoded_1 = Dense(500, activation='relu')(input_img)
    encoded_2 = Dense(8, activation='relu')(encoded_1)
    encoder = Model(input_img, encoded_2, name="encoder_model")

    # encoder.summary()

    # Decoder
    # decoder_input = tensorflow.keras.layers.Input(shape=(8), name="decoder_input")
    #
    # decoder_dense_layer1 = tensorflow.keras.layers.Dense(units=500, name="decoder_dense_1")(decoder_input)
    # decoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_dense_layer1)
    #
    # decoder_dense_layer2 = tensorflow.keras.layers.Dense(units=784, name="decoder_dense_2")(decoder_activ_layer1)
    # decoder_output = tensorflow.keras.layers.LeakyReLU(name="decoder_output")(decoder_dense_layer2)
    # decoder = tensorflow.keras.models.Model(decoder_input, decoder_output, name="decoder_model")

    decoder_input = Input(shape=(8,))
    decoded_1 = Dense(500, activation='relu')(decoder_input)
    decoded_2 = Dense(784, activation='sigmoid')(decoded_1)
    decoder = Model(decoder_input, decoded_2, name="decoder_model")
    # dncoder.summary()

    # Autoencoder
    ae_input = Input(shape=(784,), name="AE_input")
    ae_encoder_output = encoder(ae_input)
    ae_decoder_output = decoder(ae_encoder_output)

    ae = Model(ae_input, ae_decoder_output, name="AE")
    ae.summary()

    # Compile the AE model
    ae.compile(loss="mse", optimizer='adam')

    return ae,encoder,decoder

def deep_model():
    # Encoder
    input_img = Input(shape=(784,))
    encoded_1 = Dense(128, activation='relu')(input_img)
    encoded_2 = Dense(64, activation='relu')(encoded_1)
    encoded_3 = Dense(32, activation='relu')(encoded_2)
    encoder = Model(input_img, encoded_3, name="encoder_model")

    # Decoder
    decoder_input = Input(shape=(32,))
    decoded_1 = Dense(64, activation='relu')(decoder_input)
    decoded_2 = Dense(128, activation='relu')(decoded_1)
    decoded_3 = Dense(784, activation='sigmoid')(decoded_2)
    decoder = Model(decoder_input, decoded_3, name="decoder_model")

    # Autoencoder
    ae_input = Input(shape=(784,), name="AE_input")
    ae_encoder_output = encoder(ae_input)
    ae_decoder_output = decoder(ae_encoder_output)

    ae = Model(ae_input, ae_decoder_output, name="AE")
    ae.summary()

    # Compile the AE model
    ae.compile(loss="mse", optimizer='adam')

    return ae,encoder,decoder
