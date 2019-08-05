# encoding: utf8
from __future__ import unicode_literals
import re
import unicodedata
import pandas as pd 
from pandas import read_csv
import MeCab
import csv


def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

if __name__ == '__main__':
    # log print
    mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd') # Japanese dictionary path
    
    source_file = 'test_positive_1.csv'
    target_file = 'test_positive_cut1.csv'
    data_neg = read_csv(source_file,sep=',')
    #print(data_neg[0])
    print(data_neg.shape)
    #print(data_neg.head(5))
    print(data_neg.describe())
    x = data_neg.values
    y = data_neg.iloc[:,5]
    #print(x[29])
    print(x.shape)
    txt = y[0]
    print(str(txt))
    
    with open(target_file, 'w', newline='',encoding='utf-8') as csvf:
        fieldnames = ['comments']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        #writer.writeheader()
        for i in y:
            i = normalize_neologd(i)
            if pd.isna(i):
                i = '本社'
            output_words = ''
            keywords = mecab.parse(i)
            for row in keywords.split("\n"):
                word = row.split("\t")[0]
                if word == "EOS":
                    break
                else:
                    # print(word)
                    #pos = row.split("\t")[1].split(",")[0] 
                    pos = row.split("\t")[1].split(",")[1]    
                    #if pos in ['副詞', '動詞', '形容詞', '助動詞']:
                    if pos in ['自立', '*', '形容動詞語幹', '非自立', '助詞類接続', 'サ変接続', '副詞可能', '一般']:
                        output_words += str(word) + ' ' 
            writer.writerow({'comments': output_words})