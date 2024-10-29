import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_mnf.layers.mnf_dense import MNFDense


def myacc(y_true, params):
    y_pred = params[:, :1]
    delta = tf.math.abs(y_pred - y_true) / (1 + y_true)
    return tf.reduce_mean(tf.cast(delta <= 0.15, tf.float32))


def cus_neg(y_true, params):
    y_pred = params[:, :1]
    stddev = tf.math.softplus(params[:, 1:] * 0.01) + 1e-4
    negloglik = 1 / (2 * stddev**2) * tf.math.abs((y_pred -
                                                   y_true)/(1 + y_true))**2 + 1 / 2 * tf.math.log(stddev * np.sqrt(2 * np.pi))
    return negloglik


def au(y_true, params):
    # Average aleatoric Uncertainty
    stddev = 1e-4 + tf.math.softplus(params[:, 1:] * 0.01)
    return tf.reduce_mean(stddev)


def negloglik(y, params):
    rv_y = tfp.distributions.Normal(loc=params[:, :1],
                                    scale=1e-4 + tf.math.softplus(params[:, 1:] * 0.01))
    return -rv_y.log_prob(y)


def DenseFlipoutBlock(layer_in, units, train_size=30000):
    kl_divergence_fn = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) /
                        tf.cast(train_size, dtype=tf.float32))

    x = tfp.layers.DenseFlipout(
        units,
        kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
        kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
            is_singular=False),
        kernel_divergence_fn=kl_divergence_fn,
        bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
        bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
            is_singular=False),
        bias_divergence_fn=kl_divergence_fn,
    )(layer_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def base_model(model_dir):
    model = tf.keras.models.load_model(model_dir, compile=False)

    inputs = model.input
    outputs = model.layers[-8].output

    base_model = tf.keras.Model(inputs, outputs)

    return base_model


def Flipout_model(base_model, train_size):
    inputs = base_model.input
    output_layer = base_model.layers[-1].output

    for layer in base_model.layers:
        layer.trainable = False
        layer._name = layer.name + '_base'

    dense = DenseFlipoutBlock(output_layer, 256, train_size=train_size)
    dense = DenseFlipoutBlock(dense, 128, train_size=train_size)

    params = tf.keras.layers.Dense(2)(dense)

    model = tf.keras.Model(inputs, params)
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss=negloglik, metrics=[myacc, au, cus_neg])
    return model

def MNF_model(base_model):
    inputs = base_model.input
    output_layer = base_model.layers[-1].output

    for layer in base_model.layers:
        layer.trainable = False
        layer._name = layer.name + '_base'
        
    dense = MNFDense(256)(output_layer)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.LeakyReLU()(dense)

    dense = MNFDense(128)(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.LeakyReLU()(dense)

    params = tf.keras.layers.Dense(2)(dense)

    model = tf.keras.Model(inputs, params)
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss=negloglik, metrics=[myacc, au, cus_neg])
    return model


def Dropout_model(base_model, dropout_rate):
    inputs = base_model.input
    output_layer = base_model.layers[-1].output

    for layer in base_model.layers:
        layer.trainable = False
        layer._name = layer.name + '_base'

    dense = tf.keras.layers.Dense(256)(output_layer)
    dense = tf.keras.layers.Dropout(dropout_rate)(dense, training=True)
    dense = tf.keras.layers.LeakyReLU()(dense)

    dense = tf.keras.layers.Dense(128)(dense)
    dense = tf.keras.layers.Dropout(dropout_rate)(dense, training=True)
    dense = tf.keras.layers.LeakyReLU()(dense)

    params = tf.keras.layers.Dense(2)(dense)

    model = tf.keras.Model(inputs, params)
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
                  loss=negloglik, metrics=[myacc, au, cus_neg])
    return model
