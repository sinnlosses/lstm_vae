
import os
import re
import requests
import bs4
import xml.etree.ElementTree as ET
from gensim.models import KeyedVectors

class Wiki:
    def __init__(self):
        self.wiki_api_url = "http://ja.wikipedia.org/w/api.php"

    def params_get(self,query:str):
        params = {
            "format":"xml",
            "action":"query",
            "prop":"revisions",
            "titles":query,
            "rvprop":"content",
            "rvparse":""
        }
        return params

    def wiki(self, query:str):
        """wikiで検索する"""
        html = self.wiki_fetch(query)
        if html is None:
            return None
        has, redirect_word = self.has_redilect(html)
        if has:
            html = self.wiki_fetch(redirect_word)
            if html is None:
                return None
        return html

    def xml_fetch(self, url, params):
        res = requests.get(url,params=params)
        if res.status_code != 200:
            print("res_error")
            return None
        xml = ET.fromstring(res.text)
        return xml

    def wiki_fetch(self, redirect_word):
        params = self.params_get(redirect_word)
        xml = self.xml_fetch(self.wiki_api_url, params)
        if xml is None:
            return None
        try:
            html = xml[1][0][0][0][0].text
        except IndexError:
            return None
        return html

    def has_redilect(self, html_text):
        soup = bs4.BeautifulSoup(html_text, "html.parser")
        redirect = soup.find("p")
        if not redirect.get_text().startswith("転送先"):
            return (False,None)
        href = soup.find("a")
        redirect_word = re.sub(r"/wiki/","",href.get_text())
        return (True,redirect_word)

def query_get(query:str, model_path="/Users/fy/Downloads/copy/python_lab/keras/lstm/model.bin"):
    w2v = KeyedVectors.load_word2vec_format(model_path,binary=True)
    similar_list = w2v.most_similar([query])
    similar_list = [sim_word[0] for sim_word in similar_list]
    similar_list.append(query)
    return similar_list

def data_overlap_check(html_list, html):
    for h in html_list:
        if h == html:
            return False
    return True
if __name__ == '__main__':

    query = "江戸時代"
    dir_path = f"{query}_wiki_database"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    print(f"{query}の関連ワードを取得中")
    queries = query_get(query)
    print(f"{query}の関連ワードを取得しました")
    print(queries)

    client = Wiki()
    html_list = []
    for q in queries:
        print(f"{q}の処理中...")
        fname = f"{dir_path}/{q}.txt"
        html = client.wiki(q)
        if not html:
            print("fetch_failed!")
            continue
        if not data_overlap_check(html_list, html):
            print("かぶっていました")
            continue
        soup = bs4.BeautifulSoup(html, "html.parser")
        pp = soup.find_all("p")
        with open(fname, "w") as fo:
            for p in pp:
                fo.write(p.text.strip()+"\n")
        html_list.append(html)

    import pdb; pdb.set_trace()
