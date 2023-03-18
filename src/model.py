import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import InputSpec


class PConv2D(Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        self.input_dim = input_shape[0][channel_axis]

        # Image kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Mask kernel
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        # Oblicz rozmiar wypełnienia, aby osiągnąć zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
            (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
        )

        # Rozmiar okna - używany do normalizacji
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        '''
        Będziemy używać metody Keras conv2d, i zasadniczo musimy
        pomnożenie maski przez dane wejściowe X, zanim zastosujemy
        konwolucji. Dla samej maski zastosujemy konwolucje z wszystkimi wagami
        ustawionymi na 1.
        Następnie przycinamy wartości maski do wartości pomiędzy 0 a 1
        '''

        # Należy dostarczyć zarówno obraz jak i maskę
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception(
                'PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)

        # Zastosuj konwolucje do maski
        mask_output = K.conv2d(
            masks, self.kernel_mask,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Zastosuj konwolucje do obrazu
        img_output = K.conv2d(
            (images * masks), self.kernel,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Obliczanie współczynnika maski dla każdego piksela w masce wyjściowej
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Wyjście klipu ma być pomiędzy 0 a 1
        mask_output = K.clip(mask_output, 0, 1)

        # Usuń wartości współczynników, w których występują otwory
        mask_ratio = mask_ratio * mask_output

        # Normalizacja obrazu wyjściowego
        img_output = img_output * mask_ratio

        # Zastosuj bias tylko do obrazu (jeśli został wybrany)
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)

        # Zastosuj aktywacje na obrazie
        if self.activation is not None:
            img_output = self.activation(img_output)

        return [img_output, mask_output]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]


def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    """Określa długość wyjściową konwolucji dla danej długości wejściowej.
    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        stride: integer.
        dilation: dilation rate, integer.
    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = (filter_size - 1) * dilation + 1
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def __encoder_layer(filters, in_layer, in_mask):
    conv1, mask1 = PConv2D(160, (3, 3), strides=1, padding='same')([in_layer, in_mask])
    conv1 = keras.activations.relu(conv1)

    conv2, mask2 = PConv2D(160, (3, 3), strides=2, padding='same')([conv1, mask1])
    conv2 = keras.layers.BatchNormalization()(conv2, training=True)
    conv2 = keras.activations.relu(conv2)

    return conv1, mask1, conv2, mask2


def __decoder_layer(filter1, filter2, in_img, in_mask, share_img, share_mask):
    up_img = keras.layers.UpSampling2D(size=(2, 2))(in_img)
    up_mask = keras.layers.UpSampling2D(size=(2, 2))(in_mask)
    concat_img = keras.layers.Concatenate(axis=3)([share_img, up_img])
    concat_mask = keras.layers.Concatenate(axis=3)([share_mask, up_mask])

    conv1, mask1 = PConv2D(filter1, (3, 3), padding='same')([concat_img, concat_mask])
    conv1 = keras.activations.relu(conv1)

    conv2, mask2 = PConv2D(filter2, (3, 3), padding='same')([conv1, mask1])
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.activations.relu(conv2)

    return conv1, mask1, conv2, mask2


def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))


def prepare_model(hp):
    input_size = (160, 160, 3)
    input_image = keras.layers.Input(input_size, name="input_1")
    input_mask = keras.layers.Input(input_size, name="input_2")

    n_filters = hp.Int('n_filters', min_value=32, max_value=256, step=32)
    n_decoders = hp.Int('n_decoders', min_value=50, max_value=450, step=30)

    conv1, mask1, conv2, mask2 = __encoder_layer(n_filters, input_image, input_mask)
    conv3, mask3, conv4, mask4 = __encoder_layer(2 * n_filters, conv2, mask2)
    conv5, mask5, conv6, mask6 = __encoder_layer(4 * n_filters, conv4, mask4)
    conv7, mask7, conv8, mask8 = __encoder_layer(8 * n_filters, conv6, mask6)

    conv9, mask9, conv10, mask10 = __decoder_layer(8 * n_filters, 4 * n_decoders, conv8, mask8, conv7, mask7)
    conv11, mask11, conv12, mask12 = __decoder_layer(4 * n_filters, 2 * n_decoders, conv10, mask10, conv5, mask5)
    conv13, mask13, conv14, mask14 = __decoder_layer(2 * n_filters, n_decoders, conv12, mask12, conv3, mask3)
    conv15, mask15, conv16, mask16 = __decoder_layer(n_filters, 3, conv14, mask14, conv1, mask1)

    outputs = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv16)

    model = keras.models.Model(inputs=[input_image, input_mask], outputs=[outputs])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=dice_coef,
                  metrics=['accuracy'])

    return model


def make_loss_plot(history):
    sns.set()
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, loss, label='Strata trenowania', marker='o')
    plt.plot(epochs, val_loss, label='Strata walidacji', marker='o')
    plt.legend()
    plt.title('Strata trenowania i walidacji')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.show()
