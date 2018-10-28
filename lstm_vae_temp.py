
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

from utils import save_config, plot_history_loss
from utils import sub_sample, Inference, SentDataLoader, SentDataConfig

from model import LstmVae

def on_epoch_end(epoch, logs):
    if epoch % save_weight_period == 0:
        gen.save_weights(f"{weights_dir}/gen_{epoch}.hdf5")

    print('----- Generating text after Epoch: %d' % epoch)
    for _ in range(5):
        sent_surface, _ = sampling_obj.inference()
        print(sent_surface)
    return

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
    if not is_templete:
        if not os.path.exists(func_wordsets_fname):
            raise IOError(f"{func_wordsets_fname}がありません")
    if is_data_analyzed:
        if not os.path.exists(save_data_fname):
            raise IOError("保存されたanalyzed dataがありません")

    return

if __name__ == '__main__':

    # データ関係のファイルパスやディレクトリ
    data_fname = "source/copy_source.txt"
    is_templete = True
    model_dir_name = "models_temp"
    func_wordsets_fname = "func_wordsets.p"
    w2v_fname = "./model.bin"
    maxlen = 30
    mecab_lv = 4

    if is_templete:
        base_dir = "./templete_model"
    else:
        base_dir = "./language_model"

    model_dir = f"{base_dir}/{model_dir_name}"
    save_data_fname = os.path.join(model_dir, "analyzed_data.txt")
    is_data_analyzed = False
    # 活用形を情報として入れるか、品詞情報と機能語のみ使うか
    use_conjugated = True
    use_morph_func_only = False

    # 入出力データ関連
    w2v_emb_fname = os.path.join(model_dir, "emb_wordsets.p")
    save_w2i_fname = os.path.join(model_dir, "word2id.p")
    is_reversed = True

    # モデル関連
    use_loaded_emb = False
    use_loaded_model = False
    use_loaded_weight = False
    save_model_fname = os.path.join(model_dir, "model.json")
    temp_batch_size = 100
    intermediate_dim = 256
    latent_dim = 128
    kl_w = 1.0

    # callbacks系
    save_weight_period = 10
    weights_dir = os.path.join(model_dir,"weights")
    save_weights_fname = os.path.join(weights_dir, "weights.hdf5")
    save_weights_gen_fname = os.path.join(weights_dir, "gen_weights.hdf5")
    save_callback_weights_fname = os.path.join(weights_dir,"weights_{epoch:03d}_{loss:.2f}.hdf5")

    # 訓練
    initial_epoch = 0
    validation_split = 0.05
    epochs = 1000

    # その他
    save_config_fname = os.path.join(model_dir, "config.json")
    save_loss_fname = os.path.join(model_dir, "loss.png")
    data_check()

    # 解析済みデータをロードするか、解析して保存するか
    data_obj = SentDataLoader(data_fname,save_data_fname)
    if is_data_analyzed:
        sent_list = data_obj.sent_list_load(save_data_fname)
        print("解析済みデータをロードしました")
    else:
        print("データを解析します")
        sent_list = data_obj.mecabed_load(
                        use_morph_func_only=use_morph_func_only,
                        mecab_lv=mecab_lv,
                        use_conjugated=use_conjugated)
        with open(save_data_fname,"w") as fo:
            fo.write("\n".join(sent_list))

    # データの長さを制限する
    sent_list = [sent.strip() for sent in sent_list if 3 <= len(sent.split(" ")) <= maxlen]
    # validation_dataをbatch_sizeで割り切れるようにする
    if (len(sent_list)*validation_split) % temp_batch_size != 0:
        sent_list, batch_size = sub_sample(sent_list, temp_batch_size, validation_split)
    else:
        batch_size = temp_batch_size

    print("各種データの情報を取り出します")
    sent_list_config = SentDataConfig(sent_list)
    n_samples = sent_list_config.n_samples
    maxlen = sent_list_config.maxlen
    words_set = sent_list_config.create_words_set()
    # 機能語を追加する
    if not is_templete:
        with open(func_wordsets_fname,"rb") as fi:
            func_wordsets = pickle.load(fi)
        words_set = sent_list_config.add_words_set(func_wordsets)

    print("入出力データを作成中")
    X_enc, X_dec, Y, word_to_id = sent_list_config.create_ex_dx_y_w2i(is_reversed)
    # 機能語を追加する
    if not is_templete:
        word_to_id = add_word_to_id(func_wordsets)

    # モデルをロード、作成します。
    print("モデルを構築します")
    if use_loaded_emb:
        print("保存されたembをロードします")
        with open(w2v_emb_fname, "rb") as fi:
            embedding_matrix, loaded_words_set = pickle.load(fi)
        if words_set != loaded_words_set:
            raise ValueError("words_setに含まれる単語が一致しません")
    else:
        print("embを作成し、保存します")
        embedding_matrix = sent_list_config.create_emb(w2v_fname)
        with open(w2v_emb_fname,"wb") as fo:
            pickle.dump([embedding_matrix, words_set],fo)

    vae_model = LstmVae(maxlen=maxlen,
                        batch_size=batch_size,
                        intermediate_dim=intermediate_dim,
                        latent_dim=latent_dim,
                        embedding_matrix=embedding_matrix,
                        kl_w=kl_w)
    vae, enc, gen = vae_model.create_lstm_vae()
    if use_loaded_weight:
        print("重みをロードしました")
        vae.load_weights(save_weights_fname)

    vae.summary()
    save_dict = {"n_samples":n_samples,
                 "maxlen":maxlen,
                 "words_num":len(words_set),
                 "intermediate_dim":intermediate_dim,
                 "latent_dim":latent_dim,
                 "w2v_dim":len(embedding_matrix[1]),
                 "is_reversed":str(is_reversed),
                 "mecab_lv":mecab_lv,
                 "use_conjugated":str(use_conjugated)
                 }
    save_config(path=save_config_fname, save_dict=save_dict)
    with open(save_w2i_fname, "wb") as fo:
        pickle.dump([word_to_id, is_reversed], fo)

    sampling_obj = Inference(gen,
                             maxlen,
                             latent_dim,
                             word_to_id,
                             is_reversed)
    es_cb = EarlyStopping(patience=30,
                          verbose=1)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model_checkpoint = ModelCheckpoint(filepath=save_callback_weights_fname,
                                        save_weights_only=True,
                                        period=save_weight_period)

    print("Training model...")
    vae.compile(optimizer="adam", loss=vae_model.vae_loss)
    fit = vae.fit([X_enc, X_dec],Y,
             epochs=epochs,
             initial_epoch=initial_epoch,
             batch_size=batch_size,
             validation_split=validation_split,
             callbacks=[print_callback, model_checkpoint, es_cb])
    loss_history = fit.history['loss']
    val_loss_history = fit.history['val_loss']
    gen.save_weights(save_weights_gen_fname)
    vae.save_weights(save_weights_fname)
    gen_json = gen.to_json()
    with open(save_model_fname,"w") as fo:
        fo.write(gen_json)
    plot_history_loss(save_loss_fname,
                      loss_history,
                      val_loss_history=val_loss_history)
    import pdb; pdb.set_trace()
