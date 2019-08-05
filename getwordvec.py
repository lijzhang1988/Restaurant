#!/usr/bin/env python
# -*- coding: utf-8  -*-
#从词向量模型中提取文本特征向量
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import logging
import os.path
import codecs,sys
import numpy as np
import pandas as pd
import gensim
from gensim.models import word2vec
import matplotlib.pyplot as plt

# 返回特征词向量
def getWordVecs(wordList,model):
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        #print(word)
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    #print('vecs',vecs)
    return np.array(vecs, dtype=np.float32)
    

# 构建文档词向量 
def buildVecs(filename,model):
    fileVecs = []
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            #logger.info("Start line: " + line)
            wordList = line.split(' ')
            vecs = getWordVecs(wordList,model)
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean
                #print vecsArray
                #sys.exit()
                fileVecs.append(vecsArray)
            else:
                print(wordList)
                wordList = '普通'
                vecs = getWordVecs(wordList,model)
                vecsArray = sum(np.array(vecs))/len(vecs)
                fileVecs.append(vecsArray)
    return fileVecs   

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # load word2vec model
    #fdir = '/Users/sy/Desktop/pyRoot/SentimentAnalysis/'
    #inp = fdir + 'wiki.zh.text.vector'
    #model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
    
    #sentences = word2vec.Text8Corpus('2000_neg_cut_stopword.txt')
    #model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100) 

    model = word2vec.Word2Vec.load("wiki.model")

    posInput = buildVecs('two_pos_cut2.csv',model)
    negInput = buildVecs('two_neg_cut2.csv',model)
    nonInput = buildVecs('two_other_cut2.csv',model)

    # use 1 for positive sentiment， 0 for negative
    Y = np.concatenate((np.ones(len(posInput),dtype = np.int), np.zeros(len(negInput),dtype = np.int), np.linspace(2,2,len(nonInput), dtype = np.int)))
    #Y = np.concatenate((np.ones(len(posInput),dtype = np.int), np.zeros(len(negInput),dtype = np.int)))
    #Y = np.ones(len(posInput),dtype = np.int)
    #print('pos_length:',len(posInput))
    #print('neg_length:',len(negInput))
    #print('oth_length:',len(nonInput))
    #print(type(Y))
    X = posInput[:]
    
    for neg in negInput:
        X.append(neg)
    for non in nonInput:
        X.append(non)
    
    X = np.array(X)


    
    # write in file   
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    data = pd.concat([df_y,df_x],axis = 1)
    #print data
    data.to_csv('K000_data.csv')

    dataset = pd.read_csv('K000_data.csv') 
    print('data size',dataset.shape)  
    
    dataset.hist()
    plt.show() 
    
    


