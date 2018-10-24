
import re
import os
import sys
import glob
import MeCab

def wiki_text_cleaning(text:str):
    text = re.sub(r"『(.+?)』",r"「\1」",text)
    text = re.sub(r"\"(.+?)\"",r"「\1」",text)
    text = re.sub(r"”(.+?)”",r"「\1」",text)
    text = re.sub(r"\[\d+?\]","", text)
    text = re.sub(r"\(.+?\)","",text)
    text = re.sub(r"（.+?）","",text)
    text = re.sub(r"\[(.+?)\]","",text)
    text = re.sub(r",","、",text)
    text = re.sub(r"[;:]","",text)
    text = re.sub(r"。+","。",text)
    text = re.sub(r"、+","、",text)
    text = re.sub(r"「+","「",text)
    text = re.sub(r"」+","」",text)
    text = re.sub(r"'(.+?)'",r"「\1」",text)
    text = re.sub(r"[·  -]","", text)
    text = text.lower()
    text = text.replace("０","0")
    text = text.replace("１","1")
    text = text.replace("２","2")
    text = text.replace("３","3")
    text = text.replace("４","4")
    text = text.replace("５","5")
    text = text.replace("６","6")
    text = text.replace("７","7")
    text = text.replace("８","8")
    text = text.replace("９","9")

    return text

def wiki_text_structing(text:str):
    mecab = MeCab.Tagger("-Owakati")
    text = re.sub(r"。([^\n])",r"。\n\1",text)
    sent_list = text.split("\n")
    textlist = []
    for sent in sent_list:
        splited = mecab.parse(sent).strip()
        splited = splited.split(" ")
        if len(splited) > 30:
            temp = sent.split("、")
            for t in temp:
                if t != temp[-1]:
                    t += "、"
                sp = mecab.parse(t)
                sp = sp.split(" ")
                if 4 < len(sp) < 30:
                    textlist.append(t)
        else:
            textlist.append(sent)
    text = "\n".join(textlist)

    return text

if __name__ == '__main__':
    ext = ".txt"
    target_dir_name = "./江戸時代_wiki_database"
    save_dir_name = f"{target_dir_name}/save_database"
    result_path = f"{save_dir_name}/result.txt"
    if not os.path.exists(target_dir_name):
        print("targetのフォルダがありません")
        sys.exit()
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    target_fnames = glob.glob(f"{target_dir_name}/*{ext}")
    save_fnames = []
    for target in target_fnames:
        name = target.split("/")[-1].strip(ext)
        save_fnames.append(f"{save_dir_name}/{name}_clean{ext}")

    result = []
    for target, save in zip(target_fnames, save_fnames):
        with open(target, "r") as fi:
            text = fi.read()
        text = wiki_text_cleaning(text)
        text = wiki_text_structing(text)
        result.append(text)
        with open(save, "w") as fo:
            fo.write(text)

    with open(result_path,"w") as fo:
        fo.write("".join(result))
    import pdb; pdb.set_trace()
