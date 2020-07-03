#coding=utf-8
from string import punctuation
#数据预处理C:\Users\鞠甜甜\Desktop\textClassifier-master\data\rawData
import logging
import gensim
from bs4 import BeautifulSoup
import pandas as pd
with open("../unlabeledTrainData.tsv","r",encoding='utf-8') as f:
   unlabeledTrain=[line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]
with open("../labeledTrainData.tsv",encoding='utf-8') as f:
    labeledTrain=[line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 3]
#print(labeledTrain[0])

data=pd.DataFrame(labeledTrain[1:],columns=labeledTrain[0])
test=pd.DataFrame(unlabeledTrain[1:],columns=unlabeledTrain[0])
review=data['review']
label=data['sentiment']
#数据处理函数,去除HTML标签，标点符号和小写化
def dealReview(subjuct):
    #beautifulsoup函数主要用来解析html标签
    beau=BeautifulSoup(subjuct,"html.parser")#缩进格式
    newSubject=beau.get_text()#获取除html标签之外的文本
    newSubject=newSubject.replace("\\","").replace("\'","").replace('/','').replace('"', '').replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '') #去除标点符号
    newSubject=''.join([c for c in newSubject if c not in punctuation])
    newSubject=newSubject.strip().split(" ")
    newSubject=" ".join(newSubject)
    return newSubject
test=test["review"].apply(dealReview)
review=review.apply(dealReview)
#将有标签的数据和无标签的数据按字段进行合并
#保存成txt文件
review.to_csv("./train.txt",index=False,encoding='UTF-8',header=None,sep=' ')
label.to_csv("./label.txt",index=False,encoding='UTF-8',header=None,sep=' ')
new_data=pd.concat([test,review],axis=0)
new_data.to_csv("./review.txt",index=False,encoding='UTF-8',header=None,sep=' ')