
import random
import MeCab
import pickle
from gensim.models import KeyedVectors

def check(dic, word, morph):
    if not dic[word]:
        return False
    ls = dic[word]
    ls = [l.split("_")[0] for l in ls]
    if morph in ls:
        return True
    else:
        return False

if __name__ == '__main__':

    templete_path = "/Users/fy/Downloads/copy/python_lab/keras/source/copy_temp.txt"
    w2v_path = "./model.bin"
    word_morph_dict_path = "/Users/fy/Downloads/copy/python_lab/keras/others/word_morph_dict_wiki.p"

    input_word = "江戸"
    opposite_word = "東京"
    n_samples = 10
    print("loading...")
    with open(templete_path, "r") as fi:
        sent_list = fi.readlines()
        sent_list = [sent.strip() for sent in sent_list]
    sent_list = random.sample(sent_list, n_samples)

    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    with open(word_morph_dict_path,"rb") as fi:
        word_morph_dict = pickle.load(fi)

    print("comp")
    # print(sent_list)
    mecab = MeCab.Tagger()
    mecab.parse("")
    mecabed_list = []
    for sent in sent_list:
        word_morph_list = []
        node = mecab.parseToNode(sent)
        while node:
            morphes = node.feature.split(",")
            if morphes[0] == "BOS/EOS":
                node = node.next
                continue
            word = node.surface
            tail_infoes = [morphes[0]]
            tail_infoes.extend([morphes[1], morphes[5]])
            tail_infoes = "_".join(tail_infoes)
            wm = f"{word}_{tail_infoes}"
            word_morph_list.append(wm)
            node = node.next
        result = " ".join(word_morph_list)
        mecabed_list.append(result)

    # print(mecabed_list)
    # import pdb; pdb.set_trace()
    results = []
    results_surface = []
    for sent in mecabed_list:
        sentence = []
        word_list = sent.split(" ")
        sent_surface = [w.split("_")[0] for w in word_list]
        sent_surface = " ".join(sent_surface)
        results_surface.append(sent_surface)
        for word_morph in word_list:
            splited = word_morph.split("_")
            word = splited[0]
            hinshi = splited[1]
            base_form = splited[-1]
            if hinshi in ["助詞","助動詞","接続詞","記号"]:
                sentence.append(word)
                continue
            if not word in w2v.wv.vocab:
                sentence.append(word)
                continue
            trans_words = w2v.most_similar([input_word, word],[opposite_word], topn=20)
            trans_words = [w[0] for w in trans_words if word_morph_dict if check(word_morph_dict, word, hinshi)]
            if trans_words:
                if len(trans_words) >= 3:
                    trans_words = trans_words[:3]
                trans_word = random.choice(trans_words)
            else:
                trans_word = f"<{hinshi}>"
            sentence.append(trans_word)
        sentence = " ".join(sentence)
        results.append(sentence)

    with open("res.txt","w") as fo:
        for res, sur in zip(results,results_surface):
            print(f"{sur} => {res}")
            fo.write(f"{sur} => {res}\n")
    import pdb; pdb.set_trace()
