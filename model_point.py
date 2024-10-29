import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import pandas as pd
import datashader as ds
from datashader.mpl_ext import dsshow
import matplotlib.pyplot as plt
import numpy as np

def metric_sigma(z_pred, z_spec):
    del_z = z_pred - z_spec
    sigma_nmad = 1.48 * \
        np.median(np.abs((del_z - np.median(del_z))/(1 + z_spec)))
    return np.around(sigma_nmad, 3)


def metric_eta(z_pred, z_spec):
    delt_z = np.abs(z_pred - z_spec) / (1 + z_spec)
    et = np.sum((delt_z > 0.15)) / np.shape(z_pred)[0] * 100
    # et = (np.shape(np.where(delt_z > 0.15 * (1 + z_spec))[0])[0] / np.shape(z_pred)[0]) * 100
    return np.around(et, 2)


def myacc(y_true, y_pred):
    delta = tf.math.abs(y_pred - y_true) / (1 + y_true)
    return tf.reduce_mean(tf.cast(delta <= 0.15, tf.float32))


def spatial_attention(input_feature):
    kernel_size = 7

    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(
        x, axis=3, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(
        x, axis=3, keepdims=True))(input_feature)

    concat = tf.concat([avg_pool, max_pool], axis=-1)
    cbam_feature = layers.Conv2D(1, kernel_size, strides=1, padding='same',
                                 activation='sigmoid', kernel_initializer='he_normal',
                                 use_bias=False)(concat)

    return layers.multiply([input_feature, cbam_feature])


def channel_attention(input_feature, ratio=8):

    channel = input_feature.shape[-1]
    shared_layer_one = layers.Dense(channel//ratio,
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel,
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.multiply([input_feature, cbam_feature])


def cbam_attention(input_feature):
    ca = channel_attention(input_feature, ratio=8)
    sa = spatial_attention(ca)
    return sa


# class CBAM(tf.keras.layers.Layer):

#     def __init__(self, ratio=8):
#         super().__init__()
#         self.ratio = ratio

#     def call(self, x):
#         ca = channel_attention(x, self.ratio)
#         sa = spatial_attention(ca)
#         return sa



def ResNetBlock(inputs, channels, down_sample=False, kernel_size=(3, 3),
                init='he_normal'):

    __strides = [2, 1] if down_sample else [1, 1]

    x = layers.Conv2D(channels, strides=__strides[0],
                      kernel_size=kernel_size, padding='same', kernel_initializer=init)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(channels, strides=__strides[1],
                      kernel_size=kernel_size, padding='same',
                      kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    
    if down_sample:
        res = layers.Conv2D(channels, strides=2,
                            kernel_size=(1, 1), padding='same',
                            kernel_initializer=init)(inputs)
    elif inputs.shape[-1] != channels:
        res = layers.Conv2D(channels, kernel_size=1, strides=1, padding='same')(inputs)
    else:
        res = inputs

    x = layers.Add()([x, res])
    outputs = layers.ReLU()(x)
    return outputs

def median(x):
    return tfp.stats.percentile(x, 50.)

def sigma_nmad(y_true, y_pred):
    del_z = y_pred - y_true
    sigma_nmad = 1.48 * median(tf.math.abs((del_z - median(del_z))/(1 + y_true)))
    return sigma_nmad


def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    condition = tf.abs(error) < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(condition, squared_loss, linear_loss)

def loss_func(y_true, y_pred, delta=0.01):
    error = y_pred - y_true
    abs_error = tf.math.abs(error)
    quadratic = tf.math.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * tf.math.square(quadratic) + delta * linear

# def loss_func(y_true, y_pred):
#     err_1 = huber_loss(y_true, y_pred)
#     err_2 = sigma_nmad(y_true, y_pred)
#     # err_2 = tf.cast(err_2, dtype=tf.float32)
#     weight = 10.
#     value = err_1 + err_2 * weight
#     return value


def ResNet18(with_photometry=False):
    grz_inputs = tf.keras.Input(shape=(64, 64, 3))
    
    if with_photometry:
        photo_inputs = tf.keras.Input(shape=(10,))
        
        grz_feature = layers.Dense(10, activation='relu')(photo_inputs)
        grz_feature = layers.BatchNormalization()(grz_feature)
        
        grz_feature = layers.Dense(64*64, activation='relu')(grz_feature)
        grz_feature = layers.Reshape((64, 64, 1))(grz_feature)
        grz_inputs_with_photo = layers.Concatenate(axis=-1)([grz_inputs, grz_feature])
        conv_b1 = layers.Conv2D(64, (7, 7), strides=2,
                                padding='same', kernel_initializer='he_normal')(grz_inputs_with_photo)
    else:
        conv_b1 = layers.Conv2D(64, (7, 7), strides=2,
                                padding='same', kernel_initializer='he_normal')(grz_inputs)
    # conv_b1 = layers.BatchNormalization()(conv_b1)  # 32
    # conv = layers.MaxPooling2D(
    #     pool_size=(2, 2), strides=2, padding='same')(conv)
    conv_b1 = cbam_attention(conv_b1)
    res_b1 = ResNetBlock(conv_b1, 128)  # 16
    res_b1 = ResNetBlock(res_b1, 128)
    res_b1 = ResNetBlock(res_b1, 128)
    res_b1 = ResNetBlock(res_b1, 128, down_sample=True)
    #res_b1 = cbam_attention(res_b1)

    res_b1 = ResNetBlock(res_b1, 256)  # 8
    res_b1 = ResNetBlock(res_b1, 256)
    res_b1 = ResNetBlock(res_b1, 256)
    res_b1 = ResNetBlock(res_b1, 256, down_sample=True)
    #res_b1 = cbam_attention(res_b1)
    
    res_b1 = ResNetBlock(res_b1, 512)  # 4
    res_b1 = ResNetBlock(res_b1, 512)
    res_b1 = ResNetBlock(res_b1, 512)
    res_b1 = ResNetBlock(res_b1, 512, down_sample=True)
    #res_b1 = cbam_attention(res_b1)
    
    res_b1 = ResNetBlock(res_b1, 512)  # 2
    res_b1 = ResNetBlock(res_b1, 512)
    res_b1 = ResNetBlock(res_b1, 512)
    res_b1 = ResNetBlock(res_b1, 512, down_sample=True)
    
    # res_b1 = ResNetBlock(res_b1, 512)  # 2
    # res_b1 = ResNetBlock(res_b1, 512)
    # res_b1 = ResNetBlock(res_b1, 512)
    # res_b1 = ResNetBlock(res_b1, 512, down_sample=True)
    # #res_b1 = cbam_attention(res_b1)
    

    w1w2_inputs = tf.keras.Input(shape=(32, 32, 2))
    
    if with_photometry:
        w1w2_feature = layers.Dense(10, activation='relu')(photo_inputs)
        w1w2_feature = layers.BatchNormalization()(w1w2_feature)
        
        w1w2_feature = layers.Dense(32*32, activation='relu')(w1w2_feature)
        w1w2_feature = layers.Reshape((32, 32, 1))(w1w2_feature)
        w1w2_inputs_with_photo = layers.Concatenate(axis=-1)([w1w2_inputs, w1w2_feature])
    
    
        conv_b2 = layers.Conv2D(64, (7, 7), strides=2,
                                padding='same', kernel_initializer='he_normal')(w1w2_inputs_with_photo)
        
    else:
        conv_b2 = layers.Conv2D(64, (7, 7), strides=2,
                                padding='same', kernel_initializer='he_normal')(w1w2_inputs)
    # conv_b2 = layers.BatchNormalization()(conv_b2)  # 16
    # conv = layers.MaxPooling2D(
    #     pool_size=(2, 2), strides=2, padding='same')(conv)
    conv_b2 = cbam_attention(conv_b2)
    res_b2 = ResNetBlock(conv_b2, 128)  # 8
    res_b2 = ResNetBlock(res_b2, 128)
    res_b2 = ResNetBlock(res_b2, 128)
    res_b2 = ResNetBlock(res_b2, 128, down_sample=True)
    #res_b2 = cbam_attention(res_b2)

    res_b2 = ResNetBlock(res_b2, 256)  # 4
    res_b2 = ResNetBlock(res_b2, 256)
    res_b2 = ResNetBlock(res_b2, 256)
    res_b2 = ResNetBlock(res_b2, 256, down_sample=True)
    #res_b2 = cbam_attention(res_b2)
    
    res_b2 = ResNetBlock(res_b2, 512)  # 2
    res_b2 = ResNetBlock(res_b2, 512)
    res_b2 = ResNetBlock(res_b2, 512)
    res_b2 = ResNetBlock(res_b2, 512, down_sample=True)
    
    # res_b2 = ResNetBlock(res_b2, 512)  # 2
    # res_b2 = ResNetBlock(res_b2, 512)
    # res_b2 = ResNetBlock(res_b2, 512)
    # res_b2 = ResNetBlock(res_b2, 512, down_sample=True)
    #res_b2 = cbam_attention(res_b2)

    pool_b1 = layers.GlobalAveragePooling2D()(res_b1)
    pool_b2 = layers.GlobalAveragePooling2D()(res_b2)

    pool = layers.Concatenate(axis=-1)([pool_b1, pool_b2])

    dense = tf.keras.layers.Dense(256)(pool)
    dense = tf.keras.layers.ReLU()(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)

    dense = tf.keras.layers.Dense(128)(dense)
    dense = tf.keras.layers.ReLU()(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)
    
    outputs = layers.Dense(1)(dense)
    
    if with_photometry:
        model = tf.keras.Model([grz_inputs, w1w2_inputs, photo_inputs], outputs)
        
    else:
        model = tf.keras.Model([grz_inputs, w1w2_inputs], outputs)
        
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss='huber_loss', metrics=[myacc])
    
    return model

# ResNet18(with_photometry=True).summary()

def using_datashader(ax, x, y):

    df = pd.DataFrame(dict(x=x, y=y))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        vmin=0,
        vmax=35,
        norm="linear",
        aspect="auto",
        ax=ax,
    )

    plt.colorbar(dsartist)
    
    

class OutputEveryEpoch(tf.keras.callbacks.Callback):
    
    def __init__(self, test_data, test_z, base_dir):
        super().__init__()
        self.test_data = test_data
        self.test_z = test_z
        self.base_dir = base_dir
        
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f'{self.base_dir}/epochs/epoch_{epoch:03d}')
        test_pred = self.model.predict(self.test_data).reshape(-1)
        
        np.savez_compressed(f'{self.base_dir}/epochs/result_{epoch:03d}.npz',
                            pred=test_pred, true=self.test_z)
        
        sig = metric_sigma(test_pred, self.test_z)
        et = metric_eta(test_pred, self.test_z)
        
        fig, ax = plt.subplots()
        using_datashader(ax, self.test_z, test_pred)
        zmin = np.min(self.test_z)
        zmax = np.max(self.test_z)
        plt.xlim(zmin, zmax)
        plt.ylim(zmin, zmax)
        plt.xlabel('$z_{true}$')
        plt.ylabel('$z_{pred}$')
        plt.title('$\eta = $' + str(et) + '  $\sigma_{NMAD} = $' + str(sig))
        
        plt.savefig(f'{self.base_dir}/epochs/epoch_{epoch:03d}.png')
        plt.close()


def inception(input_layer, filters=(64, 128, 32, 32)):

        # Branch 1: 1x1 convolution
    branch1x1 = tf.keras.layers.Conv2D(filters[0], (1, 1), padding='same')(input_layer)
    branch1x1 = tf.keras.layers.BatchNormalization()(branch1x1)
    branch1x1 = tf.keras.layers.ReLU()(branch1x1)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch3x3 = tf.keras.layers.Conv2D(filters[1], (1, 1), padding='same')(input_layer)
    branch3x3 = tf.keras.layers.BatchNormalization()(branch3x3)
    branch3x3 = tf.keras.layers.ReLU()(branch3x3)
    branch3x3 = tf.keras.layers.Conv2D(filters[1], (3, 3), padding='same')(branch3x3)
    branch3x3 = tf.keras.layers.BatchNormalization()(branch3x3)
    branch3x3 = tf.keras.layers.ReLU()(branch3x3)

    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch3x3dbl = tf.keras.layers.Conv2D(filters[2], (1, 1), padding='same')(input_layer)
    branch3x3dbl = tf.keras.layers.BatchNormalization()(branch3x3dbl)
    branch3x3dbl = tf.keras.layers.ReLU()(branch3x3dbl)
    branch3x3dbl = tf.keras.layers.Conv2D(filters[2], (3, 3), padding='same')(branch3x3dbl)
    branch3x3dbl = tf.keras.layers.BatchNormalization()(branch3x3dbl)
    branch3x3dbl = tf.keras.layers.ReLU()(branch3x3dbl)
    branch3x3dbl = tf.keras.layers.Conv2D(filters[2], (3, 3), padding='same')(branch3x3dbl)
    branch3x3dbl = tf.keras.layers.BatchNormalization()(branch3x3dbl)
    branch3x3dbl = tf.keras.layers.ReLU()(branch3x3dbl)

    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    branch_pool = tf.keras.layers.Conv2D(filters[3], (1, 1), padding='same')(branch_pool)
    branch_pool = tf.keras.layers.BatchNormalization()(branch_pool)
    branch_pool = tf.keras.layers.ReLU()(branch_pool)

    # Concatenate the branches
    output = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3)

    return output


def inception_net():
    
    grz_inputs = tf.keras.Input(shape=(64, 64, 3))
    conv_b1 = tf.keras.layers.Conv2D(64, 7, padding='same', activation='relu')(grz_inputs)
    conv_b1 = cbam_attention(conv_b1)
    
    incep_b1 = inception(conv_b1, (64, 128, 32, 32))
    incep_b1 = tf.keras.layers.MaxPooling2D()(incep_b1) # 32
    
    incep_b1 = inception(incep_b1, (128, 256, 64, 64))
    incep_b1 = tf.keras.layers.MaxPooling2D()(incep_b1) # 16
    
    incep_b1 = inception(incep_b1, (256, 512, 128, 128))
    incep_b1 = tf.keras.layers.MaxPooling2D()(incep_b1) # 8
    
    incep_b1 = inception(incep_b1, (512, 1024, 256, 256))
    incep_b1 = tf.keras.layers.MaxPooling2D()(incep_b1) # 4
    
    incep_b1 = inception(incep_b1, (512, 1024, 256, 256))
    incep_b1 = tf.keras.layers.MaxPooling2D()(incep_b1) # 2
    
    
    w1w2_inputs = tf.keras.Input(shape=(32, 32, 2))
    conv_b2 = tf.keras.layers.Conv2D(64, 7, padding='same', activation='relu')(w1w2_inputs)
    conv_b2 = cbam_attention(conv_b2)
    
    incep_b2 = inception(conv_b2, (64, 128, 32, 32))
    incep_b2 = tf.keras.layers.MaxPooling2D()(incep_b2) # 16
    
    incep_b2 = inception(incep_b2, (128, 256, 64, 64))
    incep_b2 = tf.keras.layers.MaxPooling2D()(incep_b2) # 8
    
    incep_b2 = inception(incep_b2, (256, 512, 128, 128))
    incep_b2 = tf.keras.layers.MaxPooling2D()(incep_b2) # 4
    
    incep_b2 = inception(incep_b2, (512, 1024, 256, 256))
    incep_b2 = tf.keras.layers.MaxPooling2D()(incep_b2) # 2
    
    pool_b1 = layers.GlobalAveragePooling2D()(incep_b1)
    pool_b2 = layers.GlobalAveragePooling2D()(incep_b2)

    pool = layers.Concatenate(axis=-1)([pool_b1, pool_b2])

#     dense = tf.keras.layers.Dense(256)(pool)
#     dense = tf.keras.layers.ReLU()(dense)
#     dense = tf.keras.layers.BatchNormalization()(dense)

#     dense = tf.keras.layers.Dense(128)(dense)
#     dense = tf.keras.layers.ReLU()(dense)
#     dense = tf.keras.layers.BatchNormalization()(dense)
    
    outputs = layers.Dense(1)(pool)

    model = tf.keras.Model([grz_inputs, w1w2_inputs], outputs)
    return model