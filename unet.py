import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Dropout, concatenate

class U_Net(tf.keras.Model):

    def __init__(self):
        super(U_Net, self).__init__()
        # FILL IN CODE HERE #
        n_filters = 16

        self.conv1 = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')
        self.p1 = MaxPool2D((2, 2))
        self.d1 = Dropout(0.1)

        self.conv2 = Conv2D(filters=n_filters * 2, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')
        self.p2 = MaxPool2D((2, 2))
        self.d2 = Dropout(0.1)

        self.conv3 = Conv2D(filters=n_filters * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')
        self.p3 = MaxPool2D((2, 2))
        self.d3 = Dropout(0.1)

        self.conv31 = Conv2D(filters=n_filters * 8, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')
        self.p31 = MaxPool2D((2, 2))
        self.d31 = Dropout(0.1)

        self.mid_conv = Conv2D(filters=n_filters * 8, kernel_size = (3, 3), padding='same', kernel_initializer='he_normal')

        self.upconv32 = UpSampling2D((2, 2))
        self.d32 = Dropout(0.1)
        self.conv32 = Conv2D(filters=n_filters * 8, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')

        self.upconv1 = UpSampling2D((2, 2))
        self.d4 = Dropout(0.1)
        self.conv4 = Conv2D(filters=n_filters * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')

        self.upconv2 = UpSampling2D((2, 2))
        self.d5 = Dropout(0.1)
        self.conv5 = Conv2D(filters=n_filters * 2, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')

        self.upconv3 = UpSampling2D((2, 2))
        self.d6 = Dropout(0.1)
        self.conv6 = Conv2D(filters=n_filters * 1, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')

        self.last_conv = Conv2D(3, (1, 1), activation='sigmoid')
        # END CODE #

    def call(self, inputs):
        # FILL IN CODE HERE #
        c1 = self.conv1(inputs)
        p1 = self.p1(c1)
        d1 = self.d1(p1)

        c2 = self.conv2(d1)
        p2 = self.p2(c2)
        d2 = self.d2(p2)

        c3 = self.conv3(d2)
        p3 = self.p3(c3)
        d3 = self.d3(p3)

        c31 = self.conv31(d3)
        p31 = self.p31(c31)
        d31 = self.d31(p31)

        # print(d31.shape)

        mid_conv = self.mid_conv(d31)

        # print(mid_conv.shape)

        u32 = self.upconv32(mid_conv)
        u32 = concatenate([u32, c31])
        d32 = self.d32(u32)
        c32 = self.conv32(d32)

        u1 = self.upconv1(c32)
        u1 = concatenate([u1, c3])
        d4 = self.d4(u1)
        c4 = self.conv4(d4)

        u2 = self.upconv2(c4)
        u2 = concatenate([u2, c2])
        d5 = self.d5(u2)
        c5 = self.conv5(d5)

        u3 = self.upconv3(c5)
        u3 = concatenate([u3, c1])
        d6 = self.d6(u3)
        c6 = self.conv6(d6)

        output = self.last_conv(c6)
        # END CODE #
        return output
