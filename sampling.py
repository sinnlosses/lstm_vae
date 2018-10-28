
import os
import re
import random
import sys
import csv
import json
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, model_from_json
import pickle
from utils import Inference


if __name__ == '__main__':
    model_dir = "./templete_model/models_5000"
    model_fname = f"{model_dir}/model.json"
    weights_dir = f"{model_dir}/weights"
    # weights = "weights.hdf5"
    weights = "gen_100.hdf5"
    weights_fname = f"{weights_dir}/{weights}"
    word2id_fname = f"{model_dir}/word2id.p"
    config_json = f"{model_dir}/config.json"
    save_sample_fname = f"{model_dir}/sampling_{weights}.txt"

    with open(save_sample_fname, "w") as fo:
        pass
    with open(word2id_fname,"rb") as fi:
        word_to_id, is_reversed = pickle.load(fi)
    id_to_word = {i:w for w,i in word_to_id.items()}

    with open(config_json, "r") as fi:
        config = json.load(fi)
    with open(model_fname,"r") as fi:
        model_json = fi.read()
    gen_model = model_from_json(model_json)
    gen_model.load_weights(weights_fname)

    n_samples = 100
    maxlen = int(config["maxlen"])
    latent_dim = int(config["latent_dim"])
    sampling_obj = Inference(gen_model,
                            maxlen,
                            latent_dim,
                            word_to_id,
                            is_reversed)
    print('----- Generating text -----')
    surface_morph = []
    for n_sample in range(n_samples):
        sent_surface, sent_morph = sampling_obj.inference("random", var=3)
        print(sent_surface)
        # surface_morph.append([sent_surface,sent_morph])
        surface_morph.append(f"{n_sample}-----\n{sent_surface}\n")
    with open(save_sample_fname,"a") as fo:
        for sent in surface_morph:
            fo.write(sent)
