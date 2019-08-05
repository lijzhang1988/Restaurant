#!/usr/bin/env python
# -*- coding: utf-8  -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import logging
import os.path
import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import MeCab
from pandas import read_csv
import csv
import gensim
from gensim.models import word2vec
import time
import re
import unicodedata

def get_word(source_file, target_file):
    """Comment by space.
    Args:
        source_file: source file.
        target_file: target file(Made after participle).
    Returns:
           None. 
    """

    mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd') # Japanese dictionary path
    
    data_neg = read_csv(source_file,sep=',')
    print(data_neg.shape)
    print(data_neg.describe())
    x = data_neg.values
    y = data_neg.iloc[:,5] # Extract comment field
    print(x.shape)
    
    with open(target_file, 'w', newline='',encoding='utf-8') as csvf:
        fieldnames = ['comments']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames) # writer csv's header
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
                    pos = row.split("\t")[1].split(",")[1]    
                    if pos in ['自立', '*', '形容動詞語幹', '非自立', '助詞類接続', 'サ変接続', '副詞可能', '一般']:
                        output_words += str(word) + ' ' 
            writer.writerow({'comments': output_words})

def getWordVecs(wordList,model):
    """Return feature word vector.
    Args: 
         wordList: one line of comments.
         model: convert a single word into a vector.
    Returns:
            vector array.     
    """
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype=np.float32)
    

def buildVecs(filename,model):
    """Build document word vector.
    Args: 
         filename: File after word segmentation.
         model: convert a single word into a vector.
    Returns:
            vector list.    
    """
    fileVecs = []
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            wordList = line.split(' ')
            vecs = getWordVecs(wordList,model)
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean
                fileVecs.append(vecsArray)
            else:
                logger.info("Abnormal data vecsArray: " + str(vecsArray))
                fileVecs.append(vecsArray)
    return fileVecs   

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
    s = re.sub("[a-zA-Z0-9]","",s)
    s = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，・？?、~@#￥%……&*（）❗)‼⭐✨)。☆;)〓:❤💨д♪⁉⚡´∀`〇＝-😄🎵※-■□゜+.♪ルンヾ(●'∀'●)ノルン♪.-★�]+", "",s) 
    return s

def create_month_trend(n_years):
    date_month_sum = []
    date_month_pos_neg = {}
    
    for i in range(12 * n_years):
        w_date = time.strftime('%Y%m',time.localtime(time.time()-2592000*i))
        date_month_pos_neg = {'w_date':w_date,'positive':0,'negitive':0}
        date_month_sum.append(date_month_pos_neg)
    return date_month_sum

if __name__ == '__main__':
    # log print
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    
    # source file get word by space
    source_file = 'final_test.csv'
    target_file = 'final_test_cut.csv'
    vector_file = 'vector_data.csv'
    get_word(source_file, target_file)

    # load word2vec model
    model = word2vec.Word2Vec.load("wiki.model")

    # Convert the input Japanese into a word vector.
    vecs_input = buildVecs(target_file,model)
    X = vecs_input[:]
    X = np.array(X)
 
    df_x = pd.DataFrame(X)
    data = pd.concat([df_x],axis = 1)
    # Write the word vector as a csv document.
    data.to_csv(vector_file)
    # read in vector file.
    dataset = pd.read_csv(vector_file) 
    print('data size',dataset.shape)  
    
    fdir = ''
    df = pd.read_csv(fdir + vector_file)
    x = df.iloc[:,1:]
    
    # dimension=200
    feature_columns = [tf.contrib.layers.real_valued_column("")]

    # Construct a DNN classifier, 3 layers, each of which has 10, 20, 10 nodes, 
    # 3 target categories, and specifies the save location.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10,20,10],
                                                n_classes=3,
                                                model_dir="iris_model_v18")

    analysis_result = list(classifier.predict(x))
    logger.info('Predictions: {}'.format(str(analysis_result)))
    
    data_list = read_csv(source_file,sep=',')
    
    x = data_list.values
    print(x[0][0])
    date_month = data_list.iloc[:,3]

    other_syle_time = []
    for i in date_month:
        time_arry = time.strptime(i[0:7], '%Y-%m')
        other_syle_time.append(time.strftime('%Y%m', time_arry))

    date_month_sum = []
    date_month_pos_neg = {}
    # Extract the range of the month trend graph（years）.
    n_years = 1
    date_month_sum = create_month_trend(n_years)

    x_np = np.array(x)
    df_x = pd.DataFrame(x)
    df_y = pd.DataFrame(analysis_result)
    df_d = pd.DataFrame(other_syle_time)

    data = pd.concat([df_y,df_x,df_d],axis = 1) 
    print(data.head(5))

    data_conv = data.values
    print(data_conv[0][0:]) 
    result_dict = {}
    result_list = []
    pos_count = 0
    neg_count = 0
    other_count = 0

    shop_name = data_conv[0][1]
    
    for i in data_conv:
        if shop_name == i[1]:
            if i[0] == 1:
                for j in range(len(date_month_sum)):
                    if i[-1] == str(date_month_sum[j]['w_date']):
                        date_month_sum[j]['positive'] +=1
                pos_count += 1
            elif i[0] == 0:
                for j in range(len(date_month_sum)):
                    if i[-1] == str(date_month_sum[j]['w_date']):
                        date_month_sum[j]['negitive'] +=1
                neg_count += 1
            else:
                other_count += 1
        else:
            result_dict = {'shop_name': shop_name, 'tot_count': pos_count + neg_count + other_count, 
                           'pos_count': pos_count, 'neg_count': neg_count, 'other_count': other_count,'month_trend': date_month_sum}
            result_list.append(result_dict)
            date_month_sum = create_month_trend(n_years)
            shop_name = i[1]
            pos_count = 0
            neg_count = 0
            other_count = 0
            if i[0] == 1:
                pos_count += 1
            elif i[0] == 0:
                neg_count += 1
            else:
                other_count += 1
    result_dict = {'shop_name': shop_name, 'tot_count': pos_count + neg_count + other_count, 
                   'pos_count': pos_count, 'neg_count': neg_count, 'other_count': other_count, 'month_trend': date_month_sum}
    result_list.append(result_dict)

print('result:',result_list)

