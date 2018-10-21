# coding: utf-8

from keras import backend as K
from keras import objectives
from keras.layers import Input, LSTM, Embedding
from keras.layers.core import Dense, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.models import Model

def on_epoch_end(epoch, logs):
    BorEOS = ""
    print('----- Generating text after Epoch: %d' % epoch)
    x_pred = np.zeros(shape=(1,maxlen),dtype='int32')
    x_pred[0,0] = word_to_id[BorEOS]
    z_pred = np.random.normal(0,1,(1,intermediate_dim))
    sentence = []
    for i in range(maxlen-1):
        preds = gen.predict([x_pred,z_pred,z_pred], verbose=0)[0]
        output_id = choise_output_word_id(preds[i], mode="greedy")
        output_word = id_to_word[output_id]
        sentence.append(output_word)
        if output_word == BorEOS:
            break
        x_pred[0,i+1] = output_id
    if sentence[-1] != BorEOS:
        err_mes = "produce_failed!"
        print(err_mes)
        return

    del sentence[-1]
    sent_surface = [w_m.split("_")[0] for w_m in sentence]
    sent_surface = " ".join(sent_surface)
    print(sent_surface)

    return

def create_lstm_vae(maxlen:int,
                    batch_size:int,  # we need it for sampling
                    intermediate_dim:int,
                    latent_dim:int,
                    embedding_matrix,
                    ):
    """
    Creates an LSTM Variational Autoencoder (VAE).

    # Arguments
        maxlen: int, max length of sentences
        batch_size: int.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.
    """
    words_num = embedding_matrix.shape[0]
    w2v_dim = embedding_matrix.shape[1]
    x = Input(shape=(maxlen,), dtype='int32', name='main_input')
    emb = Embedding(input_dim=words_num,
                    output_dim=w2v_dim,
                    input_length=maxlen,
                    weights=[embedding_matrix],
                    mask_zero=True,
                    trainable=True)
    x_embed = emb(x)
    # LSTM encoding
    h = LSTM(units=intermediate_dim,
             return_sequences=False)(emb)
    # VAE Z layer
    z_mean = Dense(units=latent_dim)(h)
    z_log_sigma = Dense(units=latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    z_reweighting = Dense(units=intermediate_dim, activation="linear")
    z_reweighted = z_reweighting(z)

    # "next-word" data for prediction
    decoder_words_input = Input(shape=(maxlen,))
    decoded_x_embed = emb(decoder_words_input)
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim,
                     return_sequences=True,
                     return_state=False)

    # todo: not sure if this initialization is correct
    h_decoded = decoder_h(decoded_x_embed,
                        initial_state=[z_reweighted, z_reweighted])
    decoder_dense = TimeDistributed(Dense(words_num+1,
                                          activation="softmax",
                                          name='main_output'))
    decoded_onehot = decoder_dense(h_decoded)

    # end-to-end autoencoder
    vae = Model([x, decoder_words_input], decoded_onehot)

    # encoder, from inputs to latent space
    encoder = Model(x, [z_mean, z_log_sigma])

    # generator, from latent space to reconstructed inputs -- for inference's first step
    decoder_state_input = Input(shape=(latent_dim,))
    _z_reweighted = z_reweighting(decoder_state_input)
    _h_decoded = decoder_h(decoder_words_input,
                           initial_state=[_z_reweighted, _z_reweighted])
    _decoded_onehot = decoder_dense(_h_decoded)
    generator = Model([decoder_words_input,decoder_state_input],
                      [_decoded_onehot])

    def vae_loss(x, x_decoded_onehot):
        xent_loss = objectives.categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer="adam", loss=vae_loss)
    vae.summary()

    return vae, encoder, generator

if __name__ == '__main__':
    maxlen = 40
    batch_size = 32
    intermediate_dim = 64
    latent_dim = 50
    words_num = 1000
    embedding_matrix =
    vae, encoder, generator = create_lstm_vae(
                                maxlen,
                                batch_size,
                                intermediate_dim,
                                latent_dim,
                                words_num,
                                embedding_matrix=None)
