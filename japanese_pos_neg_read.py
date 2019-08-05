# encoding: utf8
from __future__ import unicode_literals
import re
import unicodedata
import pandas as pd 
from pandas import read_csv
import MeCab
import csv



mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd') # Japanese dictionary path

npjp = read_csv('ja_positive_negtive.csv', 
                sep=':',
                encoding='utf-8',
                names=('Tango','Yomi','Hinshi','Score'))

#print(npjp)

tango_retu = npjp['Tango']
score_retu = npjp['Score']
npjp_dic = dict(zip(tango_retu, score_retu))

source_file = 'test_negitive_0.csv'
target_file = 'test_negitive_cut1.csv'
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
        
        if pd.isna(i):
            i = '第三人称'
        output_words = ''
        output_scores = 0
        keywords = mecab.parse(i)
        for row in keywords.split("\n"):
            word = row.split("\t")[0]
            if word == "EOS":
                break
            else:
                # print(word)
                pos = row.split("\t")[1].split(",")[0] 
                #pos = row.split("\t")[1].split(",")[1]    
                if pos in ['名詞','副詞', '動詞', '形容詞', '助動詞']:
                #if pos in ['自立', '*', '形容動詞語幹', '非自立', '助詞類接続', 'サ変接続', '副詞可能', '一般']:
                    output_words += str(word) + ' '
                    if word in npjp_dic:
                        output_scores += npjp_dic[word]
                    else:
                        output_scores += 0
        writer.writerow({'comments': output_scores})
