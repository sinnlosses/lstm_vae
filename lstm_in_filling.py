
import os
import re
import random
import numpy as np
import pickle
import json
import csv

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from keras import backend as K
from keras import objectives
from keras.layers import Input, LSTM, Embedding, InputSpec
from keras.layers.core import Dense, Lambda, Dropout
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, model_from_json
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping


def infill_choice_id(distribution, morph, mode='greedy'):
    output_ids = np.argsort(distribution)[::-1]
    morph = morph.strip("<>")
    def complete_check(id):
        if id == 0:
            return False
        output = id_to_word[id].split("_")[1:]
        output = "_".join(output)
        if output != morph:
            return False
        return True

    def morph_check(id):
        if id == 0:
            return False
        output = id_to_word[id].split("_")[1]
        if output != morph.split("_")[0]:
            return False
        return True

    if mode == "greedy":
        candidate_topn = output_ids[:10]
        candidate = [c for c in candidate_topn if complete_check(c)]
        if candidate:
            output_id = candidate[0]
            return output_id

        candidate = [c for c in candidate_topn if morph_check(c)]
        if candidate:
            return candidate[0]

        candidate_topn = output_ids[10:]
        for c in candidate_topn:
            if morph_check(c):
                output_id = c
                return output_id
        return None
    else:
        raise ValueError("modeの値が間違っています")


def create_sava_dir(temp_dir, lang_dir):
    temp_dir_last_name = temp_dir.split("/")[-1]
    lang_dir_last_name = lang_dir.split("/")[-1]

    save_dir = "./{}_{}".format(temp_dir_last_name, lang_dir_last_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    return save_dir


if __name__ == '__main__':

    templete_dir = "./templete_model/models_5000_l128_h256"
    templete_csv_fname = "{}/sampling_gen_weights.hdf5.csv".format(templete_dir)
    lang_dir = "./language_model/model_coffee"
    lang_model_fname = "{}/model.json".format(lang_dir)
    lang_weights_fname = "{}/weights/gen_weights.hdf5".format(lang_dir)
    save_dir = create_sava_dir(templete_dir,lang_dir)
    save_fname = "{}/in_filling_result.txt".format(save_dir)
    word2id_fname = "{}/word2id.p".format(lang_dir)
    config_json = f"{lang_dir}/config.json"

    with open(config_json, "r") as fi:
        config = json.load(fi)

    with open(templete_csv_fname,"r") as fi:
        templete = csv.reader(fi)
        templete = [t for t in templete]

    with open(lang_model_fname,"r") as fi:
        model = model_from_json(fi.read())
    model.load_weights(lang_weights_fname)

    # n_samples for each sample
    n_samples = 10
    maxlen = int(config["maxlen"])
    latent_dim = int(config["latent_dim"])

    with open(word2id_fname, "rb") as fi:
        word_to_id, is_reversed = pickle.load(fi)
    id_to_word = {i:w for w,i in word_to_id.items()}

    # if len(templete) < n_samples:
    #     raise ValueError("サンプル数が上限を超えています\ntemplete: ".format(str(len(templete))))

    BorEOS = "<BOS/EOS>_BOS/EOS".lower()
    results = []
    for sample in templete:
        sample_csv = sample[1]
        sample_wakati_list = sample_csv.split(" ")
        results.append(f"<{sample}>")
        for _ in range(n_samples):
            is_out_of_bounds = False
            x_pred = np.zeros(shape=(1,maxlen),dtype='int32')
            x_pred[0,0] = word_to_id[BorEOS]
            z_pred = np.random.normal(0,1,(1,latent_dim))
            sentence = []
            for i in range(len(sample_wakati_list)):
                if not re.match(r"<.+>",sample_wakati_list[i]):
                    if i < maxlen-1:
                        x_pred[0,i+1] = word_to_id[sample_wakati_list[i]]
                    sentence.append(sample_wakati_list[i].split("_")[0])
                    continue
                preds = model.predict([x_pred,z_pred], verbose=0)[0]
                output_id = infill_choice_id(preds[i], morph=sample_wakati_list[i],mode="greedy")
                if output_id is None:
                    is_out_of_bounds = True
                    break
                word = id_to_word[output_id]
                if i < maxlen-1:
                    x_pred[0,i+1] = output_id
                sentence.append(word.split("_")[0])
            if is_out_of_bounds:
                continue
            if is_reversed:
                sentence = sentence[::-1]
            sentence = " ".join(sentence)
            print(sentence)
            results.append(sentence)

    with open(save_fname,"w") as fo:
        for i ,result in enumerate(results):
            if result.startswith("<"):
                result = result.strip("<>")
                fo.write(f"-----\n")
                fo.write(f"{result}\n")
                continue

            fo.write(f"{i}-----\n")
            fo.write(f"{result}\n")
