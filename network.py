from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LeakyReLU, concatenate, Conv1D, Conv2D, BatchNormalization, LSTM, MaxPool2D
import tensorflow as tf
import numpy as np

import torch
from pickle import load


def conjunction(a, b, shp):
    # For the rules we can do:
    # conj_a = a = 1
    # conj_b = b = 2
    # disj_a = c = 1
    # disj_b = d = 1

    # So the total output of the network will be 5..
    # In a more complicated problem let's say, N, implications, it is most ideal to do the following:

    # 2^(a) = b = N
    # a = c = d, so...

    # outputs = 3a + 2^(a)

    conj = keras.layers.Multiply()([tf.reshape(keras.layers.RepeatVector(2)(a), shape=(-1, shp)), b])
    return conj


def disjunction(a, b):
    disj = a + b - keras.layers.Multiply()([a, b])
    return disj


def implication(conj, disj, shp, s=9, b0=-0.5):
    imp = tf.ones(shape=tf.shape(conj)) - conj + conj * tf.reshape(keras.layers.RepeatVector(2)(disj), shape=(-1, shp))
    sig_imp = ((1 + keras.activations.exponential((s / 2))) * (keras.activations.sigmoid(s * (imp) - s / 2)) - 1) / (
            keras.activations.exponential((s / 2)) - 1)
    return sig_imp


def aggregate(imp):  # inputs=unscaled, inputs2=scaled, sp
    agg = tf.math.reduce_prod(imp, axis=1, keepdims=True)
    agg = abs(agg)
    return agg


def conjunction_gen(a, b):
    conj = keras.layers.Multiply()([tf.reshape(keras.layers.RepeatVector(2)(a), shape=(-1, 2)), b])
    return conj


def disjunction_gen(a, b):
    disj = a + b - keras.layers.Multiply()([a, b])
    return disj


def implication_gen(conj, disj, s=9, b0=-0.5):
    imp = tf.ones(shape=tf.shape(conj)) - conj + conj * tf.reshape(keras.layers.RepeatVector(2)(disj), shape=(-1, 2))
    sig_imp = ((1 + keras.activations.exponential((s / 2))) * (keras.activations.sigmoid(s * (imp) - s / 2)) - 1) / (
            keras.activations.exponential((s / 2)) - 1)
    return sig_imp


def aggregate_gen(inputs):
    agg = tf.math.reduce_prod(inputs, axis=1, keepdims=True)
    agg = abs(agg)
    # agg = -tf.math.reduce_sum(tf.math.log(inputs), axis=1, keepdims=True)
    return agg


def build_lstm(network):
    seed = network.seed
    random_normal = keras.initializers.RandomNormal(seed=seed)
    x = Input(shape=(network.x_input_size,), dtype='float')
    x_lstm = LSTM(64, input_shape=(None, 28))(x)
    x_lstm = BatchNormalization()(x_lstm)
    lstm_output = Dense(10, activation="linear", kernel_initializer=random_normal)(x_lstm)
    model = Model(inputs=x, outputs=lstm_output)
    return model

def build_generator(network):
    seed = network.seed
    random_normal = keras.initializers.RandomNormal(seed=seed)

    if network.activation == "linear":
        activation = "linear"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "elu":
        activation = "elu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "selu":
        activation = "selu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "relu":
        activation = "relu"
        kerner_initializer = keras.initializers.he_uniform(seed=seed)
    elif network.activation == "lrelu":
        activation = LeakyReLU()
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "tanh":
        activation = "tanh"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "sigmoid":
        activation = "sigmoid"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    else:
        raise NotImplementedError("Activation not recognized")

    # linear and sinus datasets
    if network.architecture == 1:
        # This will input x & noise and will output Y.
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x)
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x_output)
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x_output)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(noise)
        noise_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(noise_output)
        noise_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(noise_output)

        concat = concatenate([x_output, noise_output])
        output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    # heteroscedastic, exp and multi-modal datasets
    elif network.architecture == 2:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x)
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x_output)
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x_output)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(noise)
        noise_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(noise_output)
        noise_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(noise_output)

        concat = concatenate([x_output, noise_output])
        output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    # CA-housing and ailerons
    elif network.architecture == 3:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])

        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.fuzzy_outs, activation="sigmoid", kernel_initializer=random_normal)(output)

        shp = int(network.fuzzy_outs - int(3*network.fuzzy_outs/5))
        T = conjunction(output[:, 0:int(network.fuzzy_outs/5)], output[:, int(3*network.fuzzy_outs/5):int(network.fuzzy_outs)], shp)
        S = disjunction(output[:, int(network.fuzzy_outs/5):int(2*network.fuzzy_outs/5)], output[:,int(2*network.fuzzy_outs/5):int(3*network.fuzzy_outs/5)])

        I = implication(T, S, shp)

        A = aggregate(I)

        model = Model(inputs=[noise, x], outputs=A)

    elif network.architecture == 4:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])

        output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.fuzzy_outs, activation=activation, kernel_initializer=random_normal)(output)

        shp = int(network.fuzzy_outs - int(3*network.fuzzy_outs/5))
        T = conjunction(output[:, 0:int(network.fuzzy_outs/5)], output[:, int(3*network.fuzzy_outs/5):int(network.fuzzy_outs)], shp)
        S = disjunction(output[:, int(network.fuzzy_outs/5):int(2*network.fuzzy_outs/5)], output[:,int(2*network.fuzzy_outs/5):int(3*network.fuzzy_outs/5)])

        I = implication(T, S, shp)

        A = aggregate(I)

        model = Model(inputs=[noise, x], outputs=A)

    elif network.architecture == 5:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])

        output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)
        
    elif network.architecture == 6:

        x = Input(shape=(network.x_input_size1, network.x_input_size2, network.channel_input_size,), dtype='float')
        x_output = Conv2D(64, 4, strides=(2,2), activation=activation, kernel_initializer=kerner_initializer)(x)
        x_output = BatchNormalization()(x_output)
        x_output = MaxPool2D()(x_output)
        x_output = Conv2D(64, 4, strides=(2,2), activation=activation, kernel_initializer=kerner_initializer)(x_output)
        x_output = BatchNormalization()(x_output)
        x_output = MaxPool2D()(x_output)
        x_output = Conv2D(64, 4, strides=(2,2), activation=activation, kernel_initializer=kerner_initializer)(x_output)
        x_output = BatchNormalization()(x_output)
        x_output = MaxPool2D()(x_output)
        x_output = keras.layers.Flatten()(x_output)
        x_output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(x_output)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])

        output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(35, activation="sigmoid", kernel_initializer=random_normal)(output)
        fuzzy_outs = 35
        shp = int(fuzzy_outs - int(3*fuzzy_outs/5))
        T = conjunction(output[:, 0:int(fuzzy_outs/5)], output[:, int(3*fuzzy_outs/5):int(fuzzy_outs)], shp)
        S = disjunction(output[:, int(fuzzy_outs/5):int(2*fuzzy_outs/5)], output[:,int(2*fuzzy_outs/5):int(3*fuzzy_outs/5)])

        I = implication(T, S, shp)

        A = aggregate(I)

        model = Model(inputs=[noise, x], outputs=A)
    else:
        raise NotImplementedError("Architecture does not exist")

    return model


def build_discriminator(network):
    seed = network.seed
    random_uniform = keras.initializers.RandomUniform(seed=seed)

    if network.activation == "linear":
        activation = "linear"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "elu":
        activation = "elu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "selu":
        activation = "selu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "relu":
        activation = "relu"
        kerner_initializer = keras.initializers.he_uniform(seed=seed)
    elif network.activation == "lrelu":
        activation = LeakyReLU()
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "tanh":
        activation = "tanh"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "sigmoid":
        activation = "sigmoid"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    else:
        raise NotImplementedError("Activation not recognized")

    # linear and sinus datasets
    if network.architecture == 1:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(10, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        T = conjunction(validity[:, 0:2], validity[:, 6:10])
        S = disjunction(validity[:, 2:4], validity[:, 4:6])
        I = implication(T, S)
        A = aggregate(I)

        model = Model(inputs=[x, label], outputs=A)

    # heteroscedastic, exp and multi-modal datasets
    elif network.architecture == 2:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(10, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        T = conjunction(validity[:, 0:2], validity[:, 6:10])
        S = disjunction(validity[:, 2:4], validity[:, 4:6])

        I = implication(T, S)

        A = aggregate(I)

        model = Model(inputs=[x, label], outputs=A)

    # CA-housing and ailerons
    elif network.architecture == 3:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)
        #shp = int(network.fuzzy_outs - int(3*network.fuzzy_outs/5))
        #T = conjunction(validity[:, 0:int(network.fuzzy_outs/5)], validity[:, int(3*network.fuzzy_outs/5):int(network.fuzzy_outs)], shp)
        #S = disjunction(validity[:, int(network.fuzzy_outs/5):int(2*network.fuzzy_outs/5)], validity[:,int(2*network.fuzzy_outs/5):int(3*network.fuzzy_outs/5)])

        #I = implication(T, S, shp)

        #A = aggregate(I)


        model = Model(inputs=[x, label], outputs=validity)

    elif network.architecture == 4:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(25, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        #shp = int(network.fuzzy_outs - int(3*network.fuzzy_outs/5))
        #T = conjunction(validity[:, 0:int(network.fuzzy_outs/5)], validity[:, int(3*network.fuzzy_outs/5):int(network.fuzzy_outs)], shp)
        #S = disjunction(validity[:, int(network.fuzzy_outs/5):int(2*network.fuzzy_outs/5)], validity[:,int(2*network.fuzzy_outs/5):int(3*network.fuzzy_outs/5)])

        #I = implication(T, S, shp)

        #A = aggregate(I)


        model = Model(inputs=[x, label], outputs=validity)

    elif network.architecture == 5:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(80, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        T = conjunction(validity[:, 0:16], validity[:, 48:80])
        S = disjunction(validity[:, 16:32], validity[:, 32:48])

        I = implication(T, S)

        A = aggregate(I)

        model = Model(inputs=[x, label], outputs=A)

    elif network.architecture == 6:
        x = Input(shape=(network.x_input_size1, network.x_input_size2, network.channel_input_size,), dtype='float')
        x_output = Conv2D(64, 4, strides=(2,2), activation=activation, kernel_initializer=kerner_initializer)(x)
        x_output = BatchNormalization()(x_output)
        x_output = MaxPool2D()(x_output)
        x_output = Conv2D(64, 4, strides=(2,2), activation=activation, kernel_initializer=kerner_initializer)(x_output)
        x_output = BatchNormalization()(x_output)
        x_output = MaxPool2D()(x_output)
        x_output = Conv2D(64, 4, strides=(2,2), activation=activation, kernel_initializer=kerner_initializer)(x_output)
        x_output = BatchNormalization()(x_output)
        x_output = MaxPool2D()(x_output)
        x_output = keras.layers.Flatten()(x_output)
        print(x_output.shape)
        x_output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(x_output)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(30, activation="sigmoid", kernel_initializer=random_uniform)(concat)
        fuzzy_outs = 30
        shp = int(fuzzy_outs - int(3*fuzzy_outs/5))
        T = conjunction(validity[:, 0:int(fuzzy_outs/5)], validity[:, int(3*fuzzy_outs/5):int(fuzzy_outs)], shp)
        S = disjunction(validity[:, int(fuzzy_outs/5):int(2*fuzzy_outs/5)], validity[:,int(2*fuzzy_outs/5):int(3*fuzzy_outs/5)])

        I = implication(T, S, shp)

        A = aggregate(I)

        model = Model(inputs=[x, label], outputs=A)

    else:

        raise NotImplementedError("Architecture does not exist")

    return model
