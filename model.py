import os
import re
import random
import numpy as np
import pickle
import json

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from keras import backend as K
from keras import objectives
from keras.layers import Input, LSTM, Embedding, InputSpec, Concatenate
from keras.layers.core import Dense, Lambda, Dropout
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, model_from_json
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model


class TimestepDropout(Dropout):
    """Timestep Dropout.

    This version performs the same function as Dropout, however it drops
    entire timesteps (e.g., words embeddings in NLP tasks) instead of individual elements (features).

    # Arguments
        rate: float between 0 and 1. Fraction of the timesteps to drop.

    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`

    # Output shape
        Same as input

    # References
        - A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (https://arxiv.org/pdf/1512.05287)
    """

    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.dropout = rate

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape

class lstm_vae():
    """
    Creates an LSTM Variational Autoencoder (VAE).

    # Arguments
        maxlen: int, max length of sentences
        batch_size: int.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.
    """
    def __init__(self,
                maxlen:int,
                batch_size:int,  # we need it for sampling
                intermediate_dim:int,
                latent_dim:int,
                embedding_matrix,
                kl_w:int=1.0):
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.embedding_matrix = embedding_matrix
        self.kl_w = kl_w

    def create_lstm_vae(self):
        words_num = self.embedding_matrix.shape[0]
        w2v_dim = self.embedding_matrix.shape[1]
        x = Input(shape=(self.maxlen,), dtype='int32', name='enc_input')
        emb = Embedding(input_dim=words_num,
                        output_dim=w2v_dim,
                        input_length=self.maxlen,
                        weights=[self.embedding_matrix],
                        mask_zero=True,
                        trainable=True,
                        name='emb')
        x_embed = emb(x)
        # x_embed = Dropout(0.5)(x_embed)
        # LSTM encoding
        h = Bidirectional(LSTM(units=self.intermediate_dim,
                 return_sequences=False,
                 # dropout=0.5,
                 # recurrent_dropout=0.5,
                 name='enc_lstm'))(x_embed)
        # VAE Z layer
        self.z_mean = Dense(units=self.latent_dim, name='z_mean')(h)
        self.z_log_sigma = Dense(units=self.latent_dim, name='z_log_sigma')(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0., stddev=1.0)
            return z_mean + K.exp(0.5*z_log_sigma) * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
        z = Lambda(sampling, name='z')([self.z_mean, self.z_log_sigma])
        z_reweighting = Dense(units=self.intermediate_dim, activation="linear")
        z_reweighted = z_reweighting(z)

        # "next-word" data for prediction
        decoder_words_input = Input(shape=(self.maxlen,),
                                    dtype='int32',
                                    name='dec_input')
        decoded_x_embed = emb(decoder_words_input)
        decoded_x_embed_dropout = TimestepDropout(0.25)(decoded_x_embed)
        # decoded LSTM layer
        decoder_h = LSTM(self.intermediate_dim,
                         return_sequences=True,
                         # dropout=0.2,
                         # recurrent_dropout=0.2,
                         name='dec_lstm')

        # todo: not sure if this initialization is correct
        h_decoded = decoder_h(decoded_x_embed_dropout,
                            initial_state=[z_reweighted, z_reweighted])
        decoder_dense = TimeDistributed(Dense(words_num,
                                              activation="softmax",
                                              name='main_output'))
        decoded_onehot = decoder_dense(h_decoded)

        # end-to-end autoencoder
        vae = Model([x, decoder_words_input], decoded_onehot)

        # encoder, from inputs to latent space
        encoder = Model(x, [self.z_mean, self.z_log_sigma])

        # generator, from latent space to reconstructed inputs -- for inference's first step
        decoder_state_input = Input(shape=(self.latent_dim,))
        _z_reweighted = z_reweighting(decoder_state_input)
        _h_decoded = decoder_h(decoded_x_embed,
                               initial_state=[_z_reweighted, _z_reweighted])
        _decoded_onehot = decoder_dense(_h_decoded)
        generator = Model([decoder_words_input,decoder_state_input],
                          [_decoded_onehot])

        return vae, encoder, generator

    def vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.sparse_categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma))
        loss = xent_loss + self.kl_w*kl_loss
        return loss
