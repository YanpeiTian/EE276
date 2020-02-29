from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape,Flatten
from keras.models import Model
from keras import backend as K
from keras import regularizers,optimizers

def conv_model():

    # Encoder
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    encoder = Model(input_img, encoded, name="encoder_model")

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    # Decoder
    decoder_input = Input(shape=(4,4,8))
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(decoder_input, decoded, name="decoder_model")

    # Convolutional autoencoder
    ae_input = Input(shape=(28,28,1), name="AE_input")
    ae_encoder_output = encoder(ae_input)
    ae_decoder_output = decoder(ae_encoder_output)

    ae = Model(ae_input, ae_decoder_output, name="AE")
    ae.summary()

    ae.compile(optimizer='adam', loss='mse')

    return ae,encoder,decoder

def mixed_model():

    # Encoder
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    encoded = Dense(8, activation='relu')(x)

    encoder = Model(input_img, encoded, name="encoder_model")

    # Decoder
    decoder_input = Input(shape=(8,))
    x = Dense(32, activation='relu')(decoder_input)
    x = Dense(128, activation='relu')(x)
    x = Reshape((4,4,8))(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(decoder_input, decoded, name="decoder_model")

    # Convolutional autoencoder
    ae_input = Input(shape=(28,28,1), name="AE_input")
    ae_encoder_output = encoder(ae_input)
    ae_decoder_output = decoder(ae_encoder_output)

    ae = Model(ae_input, ae_decoder_output, name="AE")
    ae.summary()

    ae.compile(optimizer='adam', loss='mse')

    return ae,encoder,decoder

def DQN_model():

    # Encoder
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(32, (8, 8), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (4, 4),activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3),activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    encoded = Dense(8, activation='relu')(x)

    encoder = Model(input_img, encoded, name="encoder_model")

    # Decoder
    decoder_input = Input(shape=(8,))
    x = Dense(128, activation='relu')(decoder_input)
    x = Dense(1024, activation='relu')(x)
    x = Reshape((4,4,64))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (4, 4),activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (8, 8),activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (5, 5), activation='sigmoid')(x)

    decoder = Model(decoder_input, decoded, name="decoder_model")

    # Convolutional autoencoder
    ae_input = Input(shape=(28,28,1), name="AE_input")
    ae_encoder_output = encoder(ae_input)
    ae_decoder_output = decoder(ae_encoder_output)

    ae = Model(ae_input, ae_decoder_output, name="AE")
    ae.summary()

    ae.compile(optimizer='adam', loss='mse')

    return ae,encoder,decoder
