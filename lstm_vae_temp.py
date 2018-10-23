# coding: utf-8

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
from keras.layers import Input, LSTM, Embedding, InputSpec
from keras.layers.core import Dense, Lambda, Dropout
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, model_from_json
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

from utils import sent_to_surface_conjugated, max_sent_len, save_config
from utils import create_words_set, create_vae_ex_dx_y, create_emb_and_dump
from utils import plot_history_loss, choise_output_word_id, add_funcword
from utils import lstm_predict_sampling, add_sample, inference

def on_epoch_end(epoch, logs):
    if epoch % save_weight_period == 0:
        gen.save_weights(f"{weights_dir}/gen_{epoch}.hdf5")

    print('----- Generating text after Epoch: %d' % epoch)
    sent_surface, _ = inference(gen,
                             maxlen,
                             latent_dim,
                             word_to_id,
                             id_to_word,
                             is_reversed)
    print(sent_surface)
    return
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
        words_num = embedding_matrix.shape[0]
        w2v_dim = embedding_matrix.shape[1]
        x = Input(shape=(maxlen,), dtype='int32', name='enc_input')
        emb = Embedding(input_dim=words_num,
                        output_dim=w2v_dim,
                        input_length=maxlen,
                        weights=[embedding_matrix],
                        mask_zero=True,
                        trainable=True,
                        name='emb')
        x_embed = emb(x)
        # x_embed = Dropout(0.5)(x_embed)
        # LSTM encoding
        h = Bidirectional(LSTM(units=intermediate_dim,
                 return_sequences=False,
                 # dropout=0.5,
                 # recurrent_dropout=0.5,
                 name='enc_lstm'))(x_embed)
        # VAE Z layer
        self.z_mean = Dense(units=latent_dim, name='z_mean')(h)
        self.z_log_sigma = Dense(units=latent_dim, name='z_log_sigma')(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
            return z_mean + z_log_sigma * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
        z = Lambda(sampling, name='z')([self.z_mean, self.z_log_sigma])

        z_reweighting = Dense(units=intermediate_dim, activation="linear")
        z_reweighted = z_reweighting(z)

        # "next-word" data for prediction
        decoder_words_input = Input(shape=(maxlen,),
                                    dtype='int32',
                                    name='dec_input')
        decoded_x_embed = emb(decoder_words_input)
        decoded_x_embed_dropout = TimestepDropout(0.25)(decoded_x_embed)
        # decoded LSTM layer
        decoder_h = LSTM(intermediate_dim,
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
        decoder_state_input = Input(shape=(latent_dim,))
        _z_reweighted = z_reweighting(decoder_state_input)
        _h_decoded = decoder_h(decoded_x_embed,
                               initial_state=[_z_reweighted, _z_reweighted])
        _decoded_onehot = decoder_dense(_h_decoded)
        generator = Model([decoder_words_input,decoder_state_input],
                          [_decoded_onehot])

        return vae, encoder, generator

    def vae_loss(self, x, x_decoded_onehot):
        xent_loss = objectives.categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma))
        loss = xent_loss + self.kl_w*kl_loss
        return loss

    # return vae, encoder, generator

def data_check():
    # コピーのソースの確認
    if not os.path.exists(data_fname):
        raise IOError(f"{data_fname}がありません")
    # 各モデルを保存するベースDirの確認
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    # モデルのDirの確認
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # 重みを保存するDirの確認
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    if is_lang_model:
        if not os.path.exists(func_wordsets_fname):
            raise IOError(f"{func_wordsets_fname}がありません")
    if is_data_analyzed:
        if not os.path.exists(save_data_fname):
            raise IOError("保存されたanalyzed dataがありません")

    return

if __name__ == '__main__':

    data_fname = "/home/fuji/Documents/lstm/source/copy_source.txt"
    # data_fname = "./source/wiki_edojidai.txt"
    # base_dir = "templete_model"
    base_dir = "templete_model"
    # model_dir_name = "models_5000"
    model_dir_name = "model_5000_kl0start"
    func_wordsets_fname = "func_wordsets.p"
    w2v_fname = "/home/fuji/Documents/lstm/model.bin"
    maxlen = 40
    mecab_lv = 4
    # kl_w = 0.0
    kl_w = 1.0
    # kl_w = 1.0
    save_weight_period = 10
    initial_epoch = 0
    # initial_epoch = 30
    # epochs = 30
    epochs = 100
    batch_size = 32
    intermediate_dim = 128
    latent_dim = 64
    is_data_analyzed = False
    is_lang_model = False
    is_reversed = True
    use_loaded_emb = False
    use_loaded_model = False
    use_loaded_weight = False
    use_conjugated = True

    model_dir = os.path.join(base_dir,model_dir_name)
    weights_dir = os.path.join(model_dir,"weights")
    w2v_emb_fname = os.path.join(model_dir, "emb_wordsets.p")
    save_w2i_fname = os.path.join(model_dir, "word2id.p")
    save_data_fname = os.path.join(model_dir, "analyzed_data.txt")
    save_model_fname = os.path.join(model_dir, "model.json")
    save_weights_fname = os.path.join(weights_dir, "weights.hdf5")
    save_weights_gen_fname = os.path.join(weights_dir, "gen_weights.hdf5")
    save_callback_weights_fname = os.path.join(weights_dir,"weights_{epoch:03d}_{loss:.2f}.hdf5")
    save_config_fname = os.path.join(model_dir, "config.json")
    save_loss_fname = os.path.join(model_dir, "loss.png")
    save_loss_values = os.path.join(model_dir, "loss.json")
    data_check()

    # 解析済みデータをロードするならここは必要ない
    if is_data_analyzed:
        with open(save_data_fname,"r") as fi:
            sent_list = fi.readlines()
        print("解析済みデータをロードしました")
    else:
        print("データを解析します")
        with open(data_fname, "r") as fi:
            data = fi.read()
        sent_list = sent_to_surface_conjugated(
                        data,
                        save_path=save_data_fname,
                        level=mecab_lv,
                        use_conjugated=use_conjugated)
    sent_list = [sent.strip() for sent in sent_list if 3 <= len(sent.split(" ")) <= maxlen]
    if len(sent_list) % batch_size != 0:
        sent_list = add_sample(sent_list, batch_size)

    # 各種データの情報
    n_samples = len(sent_list)
    maxlen = max_sent_len(sent_list) + 1
    words_set = create_words_set(sent_list)
    if is_lang_model:
        print("機能語のセットをロードします")
        with open(func_wordsets_fname,"rb") as fi:
            funcwords_set = pickle.load(fi)
        words_set = sorted(words_set | funcwords_set)
    words_num = len(words_set)

    print("入出力データを作成中")
    X_enc, X_dec, Y, word_to_id = create_vae_ex_dx_y(
                                        sent_list,
                                        maxlen,
                                        words_num,
                                        is_reversed=is_reversed)
    if is_lang_model:
        word_to_id = add_funcword(word_to_id, funcwords_set)
        if words_num != len(word_to_id):
            print(words_num, len(word_to_id))
            raise AssertionError("words_numとword_to_idの総数が異なります")

    if use_loaded_emb:
        print("保存されたembをロードします")
        if not os.path.exists(path=w2v_emb_fname):
            raise IOError("w2vのファイルがありません")
        with open(w2v_emb_fname, "rb") as fi:
            embedding_matrix, loaded_words_set = pickle.load(fi)
        if words_set != loaded_words_set:
            raise ValueError("words_setに含まれる単語が一致しません")
    else:
        print("embを作成し、保存します")
        embedding_matrix = create_emb_and_dump(w2v_fname,
                                               words_set,
                                               word_to_id,
                                               w2v_emb_fname)

    w2v_dim = len(embedding_matrix[1])
    id_to_word = {i:w for w,i in word_to_id.items()}
    if use_loaded_model:
        raise IOError("現在、モデルはロードできません。")
        print("保存されたモデルをロードします")
        with open(save_model_fname,"r") as fi:
            json_string = fi.read()
            vae, enc, gen = model_from_json(json_string)
    else:
        print("モデルを構築します")
        vae_model = lstm_vae(
                            maxlen=maxlen,
                            batch_size=batch_size,
                            intermediate_dim=intermediate_dim,
                            latent_dim=latent_dim,
                            embedding_matrix=embedding_matrix,
                            kl_w=kl_w)
        vae, env, gen = vae_model.create_lstm_vae()
    if use_loaded_weight:
        print("重みをロードしました")
        vae.load_weights(save_weights_fname)

    vae.summary()
    # import pdb; pdb.set_trace()
    # gen.summary()
    save_dict = {"n_samples":n_samples,
                 "maxlen":maxlen,
                 "words_num":words_num,
                 "intermediate_dim":intermediate_dim,
                 "latent_dim":latent_dim,
                 "w2v_dim":w2v_dim,
                 "is_reversed":str(is_reversed),
                 "mecab_lv":mecab_lv,
                 "use_conjugated":str(use_conjugated)
                 }
    save_config(path=save_config_fname, save_dict=save_dict)
    with open(save_w2i_fname, "wb") as fo:
        pickle.dump([word_to_id, is_reversed], fo)

    es_cb = EarlyStopping(patience=30,
                          verbose=1,
                          mode='auto')
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model_checkpoint = ModelCheckpoint(filepath=save_callback_weights_fname,
                                        save_weights_only=True,
                                        period=save_weight_period)

    print("Training model...")
    vae.compile(optimizer="adam", loss=vae_model.vae_loss)
    fit = vae.fit([X_enc, X_dec],
             Y,
             epochs=epochs,
             initial_epoch=initial_epoch,
             verbose=1,
             batch_size=batch_size,
             validation_split=0.05,
             callbacks=[print_callback, model_checkpoint, es_cb])
    loss_history = fit.history['loss']
    val_loss_history = fit.history['val_loss']
    gen.save_weights(save_weights_gen_fname)
    vae.save_weights(save_weights_fname)
    gen_json = gen.to_json()
    with open(save_model_fname,"w") as fo:
        fo.write(gen_json)
    with open(save_loss_values, "w") as fo:
        json.dump({"loss":loss_history,
                   "val_loss":val_loss_history}, fo)
    plot_history_loss(save_loss_fname,loss_history, val_loss_history=val_loss_history)
    import pdb; pdb.set_trace()
