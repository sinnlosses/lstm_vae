
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

class SentDataLoader(object):
    def __init__(self,
                 data_fname:str,
                 save_data_fname:str):
        self.mecab = MeCab.Tagger()
        self.mecab.parse("")
        self.data_fname = data_fname
        self.save_data_fname = save_data_fname
        self.BorEOS = "<BOS/EOS>_BOS/EOS".lower()

    def sent_list_load(self, data_fname):
        """与えられたファイルパスを開き、一行を一要素としてリストにして返す
            返された値は\nを含む
        """
        with open(data_fname,"r") as fi:
            sent_list = fi.readlines()
            sent_list = [sent.strip() for sent in sent_list]
        return sent_list

    def mecabed_load(self,
                     use_morph_func_only:bool=False,
                     mecab_lv:int=4,
                     use_conjugated:bool=True):
        """データファイルを読み込み表層形と付加情報にしてリストで返す。
            use_morph_func_onlyの場合、内容語を品詞に変換する。
        """
        def w_m(word, morph, tail_infoes, use_morph_func_only):
            wm = f"{word}_{tail_infoes}"
            if use_morph_func_only:
                if morph not in ["助動詞","助詞","接続詞","記号"]:
                    wm = f"{morph}_{tail_infoes}"
            return wm

        sent_list = self.sent_list_load(self.data_fname)
        mecabed_list = []
        for sent in sent_list:
            word_morph_list = []
            node = self.mecab.parseToNode(sent)
            while node:
                morphes = node.feature.split(",")
                word = node.surface
                morph = morphes[0]
                if morph == "BOS/EOS":
                    word_morph_list.append(self.BorEOS)
                    node = node.next
                    continue
                tail_infoes = morphes[:mecab_lv]
                if use_conjugated:
                    tail_infoes.append(morphes[5])
                tail_infoes = "_".join(tail_infoes)
                wm = w_m(word, morph, tail_infoes, use_morph_func_only)
                word_morph_list.append(wm)
                node = node.next
            mecabed_list.append(" ".join(word_morph_list))
        return mecabed_list

class SentDataConfig(object):
    def __init__(self, sent_list:list):
        self.sent_list = sent_list
        self.n_samples = len(self.sent_list)
        self.maxlen = max([len(sent.split(" ")) for sent in self.sent_list])
        self.words_set = None
        self.word_to_id = None
        self.funcwords_set = None

    def create_words_set(self):
        """単語のセットを返す。言語モデルなら機能語を加えて返す
        """
        sentences = " ".join(self.sent_list)
        words_sequence = text_to_word_sequence(text=sentences,
                                          filters='\n',
                                          split=" ")
        self.words_set = set(sorted(words_sequence))
        return self.words_set

    def add_words_set(self, wordsets_to_add):
        """単語のセットをwords_setに追加する
        """
        if not self.words_set:
            raise ValueError("words_setがNoneです")
        self.words_set = sorted(self.words_set | wordsets_to_add)
        return self.words_set

    def create_ex_dx_y_w2i(self,
                           is_reversed:bool=True):
        """VAEに必要なEncoder X, Decoder X, Yを出力する
            is_reversed: 逆順かどうか
        """
        if not self.maxlen:
            raise ValueError("maxlenがセットされていません")
        tokenizer = Tokenizer(filters='\n')
        tokenizer.fit_on_texts(self.sent_list)
        sent_seq = tokenizer.texts_to_sequences(self.sent_list)
        self.word_to_id = tokenizer.word_index

        if is_reversed:
            X_enc = [sent[-2:0:-1] for sent in sent_seq]
            X_dec = [sent[-1:0:-1] for sent in sent_seq]
            y_seq = [sent[-2::-1] for sent in sent_seq]
        else:
            X_enc = [sent[1:-1] for sent in sent_seq]
            X_dec = [sent[:-1] for sent in sent_seq]
            y_seq = [sent[1:] for sent in sent_seq]

        X_enc = sequence.pad_sequences(sequences=X_enc,
                                   maxlen=self.maxlen,
                                   padding='post')
        X_dec = sequence.pad_sequences(sequences=X_dec,
                                   maxlen=self.maxlen,
                                   padding='post')
        y_seq = sequence.pad_sequences(sequences=y_seq,
                                       maxlen=self.maxlen,
                                       padding='post')
        # Y = np.zeros((n_samples, maxlen, words_num+1), dtype=np.bool)
        # for i, id_seq in enumerate(y_seq):
        #     for j, id in enumerate(id_seq):
        #         if id == 0:
        #             continue
        #         Y[i,j,id] = 1.
        return X_enc, X_dec, y_seq, self.word_to_id

    def add_word_to_id(self, wordsets_to_add):
        """機能語をword_to_idに追加する
        """
        if not self.words_set:
            raise ValueError("words_setがNoneです")

        total_ids = len(self.word_to_id)
        for word in wordsets_to_add:
            if word not in self.word_to_id:
                total_ids += 1
                self.word_to_id[word] = total_ids

        if len(self.words_set) != len(self.word_to_id):
            raise AssertionError("words_numとword_to_idの総数が異なります")

        return self.word_to_id

    def create_emb(self,w2v_fname:str):
        """embedding_matrixを作成する
        """
        words_num = len(self.words_set)
        w2v = KeyedVectors.load_word2vec_format(w2v_fname, binary=True)
        w2v_dim = w2v.vector_size
        embedding_matrix = np.zeros((words_num+1, w2v_dim))
        for word in self.words_set:
            word_surface = word.split("_")[0]
            id = self.word_to_id[word]
            if word_surface not in w2v.vocab:
                embedding_matrix[id] = np.random.normal(0,1,(1,w2v_dim))
            else:
                embedding_matrix[id] = w2v.wv[word_surface]
        return embedding_matrix

def save_config(path:str, save_dict:dict):
    """
        sava_config
    """
    with open(path, "w") as fo:
        json.dump(save_dict, fo)
    return

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


def printing_sample(csv:str, save_path:str="temp.txt"):
    with open(csv, "r") as fi:
        sent_list = fi.readlines()

    sent_list = [sent.strip().split(",")[0] for sent in sent_list]
    with open(save_path, "w") as fo:
        fo.write("\n".join(sent_list))

class Inference(object):
    def __init__(self,
                 gen_model,
                 maxlen:int,
                 latent_dim:int,
                 word_to_id:dict,
                 is_reversed:bool=True):
        self.gen_model = gen_model
        self.maxlen = maxlen
        self.latent_dim = latent_dim
        self.word_to_id = word_to_id
        self.is_reversed = is_reversed
        self.id_to_word = {i:w for w,i in word_to_id.items()}
        self.BorEOS = "<BOS/EOS>_BOS/EOS".lower()

    def choise_output_word_id(self, distribution, mode='greedy'):
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
            output_ids = output_ids[:3]
            output_words = [self.id_to_word[id] for id in output_ids if id != 0]
            if self.BorEOS in output_words:
                output_id = self.word_to_id[self.BorEOS]
            else:
                output_id = random.choice(output_ids)
        else:
            raise ValueError("modeの値が間違っています")

        return output_id

    def inference(self, mode="greedy", var:int=1):
        x_pred = np.zeros(shape=(1,self.maxlen),dtype='int32')
        x_pred[0,0] = self.word_to_id[self.BorEOS]
        z_pred = np.random.normal(0,var,(1,self.latent_dim))
        sentence = []
        for i in range(self.maxlen-1):
            preds = self.gen_model.predict([x_pred,z_pred], verbose=0)[0]
            output_id = self.choise_output_word_id(preds[i], mode=mode)
            output_word = self.id_to_word[output_id]
            sentence.append(output_word)
            if output_word == self.BorEOS:
                break
            x_pred[0,i+1] = output_id
        if sentence[-1] != self.BorEOS:
            err_mes = "produce_failed!"
            return err_mes, None

        del sentence[-1]
        sent_surface = [w_m.split("_")[0] for w_m in sentence]
        if self.is_reversed:
            sent_surface = [word for word in reversed(sent_surface)]
        sent_surface = " ".join(sent_surface)
        sent_morph = [self.create_sent_morph(w_m) for w_m in sentence]
        sent_morph = " ".join(sent_morph)

        return sent_surface, sent_morph

    def create_sent_morph(self, w_m):
        word, morph = w_m.split("_")[0], w_m.split("_")[1]
        if morph in ["助詞","助動詞","記号","接続詞"]:
            res = w_m
        else:
            m = w_m.split("_")[1:]
            m = "_".join(m)
            res = "<{}>".format(m)
        return res

def sub_sample(sent_list:list, batch_size:int, validation_split:int):
    n_samples = len(sent_list)
    remainder = n_samples % batch_size
    total = n_samples - remainder
    batch_size = int(total*validation_split)

    return sent_list[:total], batch_size
