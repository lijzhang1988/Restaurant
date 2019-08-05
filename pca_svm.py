#!/usr/bin/env python
# -*- coding: utf-8  -*-
# PCA  SVM
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
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
from pandas import set_option
#import read_data



# 获取数据 [9000 rows x 200 columns]
fdir = ''
df = pd.read_csv(fdir + 'K000_data.csv')
y = df.iloc[:,1]
x = df.iloc[:,2:]
# 数据相关性
#set_option('display.width', 100)
#set_option('precision', 2)
#print(df.corr(method='pearson'))

# 数据分布情况
#print(df.skew())


#print(y)
df = pd.read_csv(fdir + '1000_data.csv')
y_t = df.iloc[:,1]
x_t = df.iloc[:,2:]
# PCA降维
##计算全部贡献率
n_components = 200
pca = PCA(n_components=n_components)
pca.fit(x)
#pca.fit(x_t)
#x = pca.transform(x)
#x_t = pca.transform(x_t)
print('x.pca:',x.shape)
print('x_t.pca:',x_t.shape)
#print pca.explained_variance_ratio_

##PCA作图
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.show()


# 分割数据为训练数据和测试数据
X_train,X_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=None,stratify=y)
#print('x_train:',X_train)
#print('x_test:',X_test)
#print('y_train:',y_train)
#print('y_test:',y_test)

'''
# K近领算法
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = KNeighborsClassifier()
result = cross_val_score(model, x, y, cv=kfold)
print('KFold:',result.mean())

# 贝叶斯分类器
model = GaussianNB()
result = cross_val_score(model, x, y, cv=kfold)
print('GaussianNB:',result.mean())

# 分类与会归树
model = DecisionTreeClassifier()
result = cross_val_score(model, x, y, cv=kfold)
print('TREE:',result.mean())

# 支持向量
model = SVC(C = 0.5, probability = True,kernel='rbf')
#result = cross_val_score(model, x, y, cv=kfold)
model.fit(X_train,y_train)
#print('SVC:',result.mean())
print('SVC:',model.score(X_test,y_test))
'''
# 调参数
'''
parameters={'kernel':['linear','rbf','sigmoid','poly'],'C':np.linspace(0.1,20,50),'gamma':np.linspace(0.1,20,20)}
svc = svm.SVC()
model = GridSearchCV(svc,parameters,cv=5,scoring='accuracy')
model.fit(X_train,y_train)
model.best_params_
print(model.score(X_test,y_test))
'''

# 深度神经网络
#training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename='4000_data.csv',target_dtype=np.int,features_dtype=np.float64)

# 每行数据4个特征，都是real-value的 dimension=200
feature_columns = [tf.contrib.layers.real_valued_column("")]

# 构建一个DNN分类器，3层，其中每个隐含层的节点数量分别为10，20，10，目标的分类3个，并且指定了保存位置
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir="iris_model_v18")

# 指定数据，以及训练的步数
classifier.fit(x=X_train,
               y=y_train,
               steps=2000)

#y1 = list(classifier.predict(x_t))
#print('Predictions: {}'.format(str(y1)))
#print('Predic_test: {}'.format(str(y_t)))
accuracy_score = classifier.evaluate(x=X_test,
                                     y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))


##根据图形取100维
#x_pca = PCA(n_components = 100).fit_transform(X_train)
#x_pcat = PCA(n_components = 100).fit_transform(X_test)

# SVM (RBF)
# using training data with 100 dimensions
'''
clf = svm.SVC(C = 50, kernel='rbf', probability = True)
clf.fit(X_train,y_train)

joblib.dump(clf,'svm_model_v9.pkl',compress=3)
#print(clf.score(test_vec,y_test))
#clf = joblib.load('svm_model.pkl')
#result = clf.predict(sent_cut_vec)

print ('Test Accuracy: %.2f'% clf.score(X_test,y_test))
'''
'''
#Create ROC curve
pred_probas = clf.predict_proba(x)[:,1] #score

fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.show()
'''

'''
#对单个句子进行情感判断
def svm_predict(sent):
    model = word2vec.Word2Vec.load('E:/NLP/chinese-w2v-sentiment/train_model.model')
    sent_cut = jieba.lcut(sent)
    sent_cut_vec = get_sent_vec(300,sent_cut,model)
    clf = joblib.load('E:/NLP/chinese-w2v-sentiment/svm_model/svm_model.pkl')
    result = clf.predict(sent_cut_vec)

    if int(result[0] == 1):
        print(sent,'pos')
    else:
        print(sent,'neg')

# 测试
#clf = joblib.load('svm_model.pkl')
#result = clf.predict(sent_cut_vec)

#情感正式开始预测
sent = '破手机，垃圾'
svm_predict(sent)
'''
