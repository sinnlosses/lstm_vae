
import os
import numpy as np
import MeCab
import pickle
import random
import json

from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.utils import to_categorical
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

def sent_to_surface_conjugated(sentences:str,
                               save_path:str,
                               level:int=4,
                               use_conjugated:bool=True):
    """
        \nで区切られた各文をmecabで解析し表層形+levelで決めた数の情報を
        アンダースコアでつなげて保存する
    """
    mecab = MeCab.Tagger()
    mecab.parse("")
    sent_list = sentences.split("\n")
    mecabed_list = []
    for sent in sent_list:
        word_morph_list = []
        node = mecab.parseToNode(sent)
        while node:
            morphes = node.feature.split(",")
            if morphes[0] == "BOS/EOS":
                wm = "<BOS/EOS>_BOS/EOS".lower()
                word_morph_list.append(wm)
                node = node.next
                continue
            word = node.surface
            tail_infoes = morphes[:level]
            if use_conjugated:
                tail_infoes.append(morphes[5])
            tail_infoes = "_".join(tail_infoes)
            wm = f"{word}_{tail_infoes}"
            word_morph_list.append(wm)
            node = node.next
        result = " ".join(word_morph_list)
        mecabed_list.append(result)

    with open(save_path, "w") as fo:
        fo.write("\n".join(mecabed_list))
    return mecabed_list

def max_sent_len(sent_list:list):
    return max([len(sent.split(" ")) for sent in sent_list])

def save_config(path:str, save_dict:dict):
    """
        sava_config
    """
    with open(path, "w") as fo:
        json.dump(save_dict, fo)
    return

def create_words_set(sent_list:list):
    sentences = " ".join(sent_list)
    words_sequence = text_to_word_sequence(text=sentences,
                                      filters='\n',
                                      split=" ")
    return set(sorted(words_sequence))

def create_vae_ex_dx_y(
                    sent_list:list,
                    maxlen:int,
                    words_num:int,
                    is_reversed:bool=False):
    """VAEに必要なEncoder X, Decoder X, Yを出力する
        sent_list: リスト
        maxlen: 最大長
        words_num: 単語の総数
        is_reversed: 逆順かどうか
    """
    n_samples = len(sent_list)
    tokenizer = Tokenizer(filters='\n')
    tokenizer.fit_on_texts(sent_list)
    sent_seq = tokenizer.texts_to_sequences(sent_list)
    word_to_id = tokenizer.word_index

    if is_reversed:
        X_enc = [sent[-2:0:-1] for sent in sent_seq]
        X_dec = [sent[-1:0:-1] for sent in sent_seq]
        y_seq = [sent[-2::-1] for sent in sent_seq]
    else:
        X_enc = [sent[1:-1] for sent in sent_seq]
        X_dec = [sent[:-1] for sent in sent_seq]
        y_seq = [sent[1:] for sent in sent_seq]

    X_enc = sequence.pad_sequences(sequences=X_enc,
                               maxlen=maxlen,
                               padding='post')
    X_dec = sequence.pad_sequences(sequences=X_dec,
                               maxlen=maxlen,
                               padding='post')
    y_seq = sequence.pad_sequences(sequences=y_seq,
                                   maxlen=maxlen,
                                   padding='post')
    Y = np.zeros((n_samples, maxlen, words_num+1), dtype=np.bool)
    for i, id_seq in enumerate(y_seq):
        for j, id in enumerate(id_seq):
            if id == 0:
                continue
            Y[i,j,id] = 1.

    # with open(fname, "wb") as fo:
    #     pickle.dump([X,Y,word_to_id],fo)
    return X_enc, X_dec, Y, word_to_id

def create_emb_and_dump(w2v_fname,
                        words_set,
                        word_to_id,
                        fname="emb_wordsets.p"):

    print("w2vデータをload中...")
    words_num = len(words_set)
    w2v = KeyedVectors.load_word2vec_format(w2v_fname, binary=True)
    w2v_dim = w2v.vector_size
    embedding_matrix = np.zeros((words_num+1, w2v_dim))
    for word in words_set:
        word_surface = word.split("_")[0]
        id = word_to_id[word]
        if word_surface not in w2v.vocab:
            embedding_matrix[id] = np.random.normal(0,1,(1,w2v_dim))
        else:
            embedding_matrix[id] = w2v.wv[word_surface]
    with open(fname,"wb") as fo:
        pickle.dump([embedding_matrix, words_set],fo)

    return embedding_matrix

def plot_history_loss(save_path:str,
                      loss_history:list,
                      val_loss_history=None):
    # Plot the loss in the history
    fig = plt.figure(1)
    plt.plot(loss_history,label="loss for training")
    if val_loss_history is not None:
        plt.plot(val_loss_history,label="loss for validation")
    plt.title("model_loss")
    plt.xlabel("epoch")
    plt.ylabel('loss: cross_entropy')
    plt.legend(loc='upper right')
    fig.savefig(save_path)

    return

def choise_output_word_id(distribution, word_to_id, id_to_word, mode='greedy'):
    BorEOS = "<BOS/EOS>_BOS/EOS".lower()
    output_ids = np.argsort(distribution)[::-1]
    def check(id):
        if id == 0:
            return False
        return True

    if mode == "greedy":
        i = 0
        while True:
            output_id = output_ids[i]
            if output_id != 0:
                break
            else:
                i += 1
    elif mode == "random":
        output_ids = output_ids[:5]
        output_words = [id_to_word[id] for id in output_ids if id != 0]
        if BorEOS in output_words:
            output_id = word_to_id[BorEOS]
        else:
            output_id = random.choice(output_ids)
    else:
        raise ValueError("modeの値が間違っています")

    return output_id

def add_funcword(word_to_id:dict, funcwords_set:set):
    total_ids = len(word_to_id)
    for func_word in funcwords_set:
        if func_word not in word_to_id:
            total_ids += 1
            word_to_id[func_word] = total_ids
    return word_to_id

def printing_sample(csv:str, save_path:str="temp.txt"):
    with open(csv, "r") as fi:
        sent_list = fi.readlines()

    sent_list = [sent.strip().split(",")[0] for sent in sent_list]
    with open(save_path, "w") as fo:
        fo.write("\n".join(sent_list))

def lstm_predict_sampling(model,
                     maxlen:int,
                     word_to_id:dict,
                     id_to_word:dict,
                     h_length:int,
                     is_reversed:bool):

    BorEOS = "<BOS/EOS>_BOS/EOS".lower()
    x_pred = np.zeros(shape=(1,maxlen),dtype='int32')
    x_pred[0,0] = word_to_id[BorEOS]
    h_pred = np.random.normal(0,1,(1,h_length))
    c_pred = np.random.normal(0,1,(1,h_length))
    sentence = []
    for i in range(maxlen-1):
        preds = model.predict([x_pred,h_pred,c_pred], verbose=0)[0]
        output_id = choise_output_word_id(preds[i], mode="greedy")
        output_word = id_to_word[output_id]
        sentence.append(output_word)
        if output_word == BorEOS:
            break
        x_pred[0,i+1] = output_id
    if sentence[-1] != BorEOS:
        err_mes = "produce_failed!"
        print(err_mes)

    del sentence[-1]
    if not sentence:
        err_mes = "white"
        print(err_mes)

    sent_surface = [w_m.split("_")[0] for w_m in sentence]
    if is_reversed:
        sent_surface = [word for word in reversed(sent_surface)]
    sent_surface = " ".join(sent_surface)
    sent_morph = [create_sent_morph(w_m) for w_m in sentence]
    sent_morph = " ".join(sent_morph)

    return (sent_surface, sent_morph)

def inference(gen_model,
              maxlen:int,
              latent_dim:int,
              word_to_id:dict,
              id_to_word:dict,
              is_reversed:bool=True):
    BorEOS = "<BOS/EOS>_BOS/EOS".lower()
    x_pred = np.zeros(shape=(1,maxlen),dtype='int32')
    x_pred[0,0] = word_to_id[BorEOS]
    z_pred = np.random.normal(0,1,(1,latent_dim))
    sentence = []
    for i in range(maxlen-1):
        preds = gen_model.predict([x_pred,z_pred], verbose=0)[0]
        output_id = choise_output_word_id(preds[i], word_to_id, id_to_word, mode="random")
        output_word = id_to_word[output_id]
        sentence.append(output_word)
        if output_word == BorEOS:
            break
        x_pred[0,i+1] = output_id
    if sentence[-1] != BorEOS:
        err_mes = "produce_failed!"
        return err_mes, None

    del sentence[-1]
    sent_surface = [w_m.split("_")[0] for w_m in sentence]
    if is_reversed:
        sent_surface = [word for word in reversed(sent_surface)]
    sent_surface = " ".join(sent_surface)
    sent_morph = [create_sent_morph(w_m) for w_m in sentence]
    sent_morph = " ".join(sent_morph)

    return sent_surface, sent_morph

def create_sent_morph(w_m):
    word, morph = w_m.split("_")[0], w_m.split("_")[1]
    if morph in ["助詞","助動詞","記号","接続詞"]:
        res = w_m
    else:
        m = w_m.split("_")[1:]
        m = "_".join(m)
        res = "<{}>".format(m)
    return res

def add_sample(sent_list:list, batch_size:int, validation_split=None):
    amari = len(sent_list) % batch_size
    num_adds = batch_size - amari
    adds = random.sample(sent_list, k=num_adds)
    sent_list.extend(adds)
    batch_size = int(len(sent_list)*validation_split)

    return sent_list, batch_size
