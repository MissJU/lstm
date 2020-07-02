import sys
#sys.setdefaultencoding('utf8')
import collections
import numpy
from motion_classify import utils
"""
创建词汇表
:return:
"""
with open('./review.txt','r',encoding="mac_roman") as f:
    reviews=f.read()
#print(reviews)
with open('./label.txt','r') as f:
    labels=f.read()
with open('./train.txt','r',encoding="mac_roman") as f:
    train_reviews=f.read()

from string import punctuation
#移除所有标点符号
all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')
train_text=''.join([c for c in train_reviews if c not in punctuation])
train_reviews=train_text.split('\n')
#print(len(reviews))
#print(len(train_reviews))
all_text = ' '.join(reviews)
#文本分为单词列表
words = all_text.split()

#创建数字字典，将评论对应转换成整数，输入embedding层
from collections import Counter
count = Counter(words)

#按计数进行排序
vocab_size=100000
vocab = sorted(count,key=count.get,reverse=True)
#print(vocab)
#选取高频词
#word_list=[word[0] for word in vocab]
print(len(vocab))
word_list=['<unkown>']+vocab[:vocab_size-1]

# #将词汇表写入文件中
# with open('./vocab.txt','w') as f:
#     for word in word_list:
#         f.write(word+'\n')

# 生成字典：{单词：整数}
vocab2id = {word:i for i,word in enumerate(word_list,1)}

#将语句转化成向量

def get_id_by_word(vocab,vocab2id):
    if vocab in vocab2id.keys():
        return vocab2id[vocab]
    else:
        return vocab2id['<unkown>']
reviews_ints = []
for each in train_reviews:
    reviews_ints.append([str(get_id_by_word(word,vocab2id)) for word in each.split()])

labels=[int(label) for label in labels if label!='\n']
#统计已转成词的句子的长度
from collections import Counter
review_lens = Counter([len(x) for x in reviews_ints])


for x in  reviews_ints:
    if len(x)<=1:
        print(x)
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))
#print(review_lens)
#将0长度评论删除
#找出非0长度列表的索引
import numpy as np
non_zero_idx = [i for i,review in enumerate(reviews_ints) if len(review)>0]
reviews_ints = [reviews_ints[i] for i in non_zero_idx]
labels = [labels[i] for i in non_zero_idx]
print(labels,len(labels))
#print(labels)
data=np.zeros((len(reviews_ints),max(review_lens)),dtype=int)
print(data.shape)
for row in range(len(reviews_ints)):
    data[row,:len(reviews_ints[row])]=reviews_ints[row]

with open('./vec.txt','w') as f:
    for vec in reviews_ints:
        #print(2)
        f.write(' '.join(vec)+'\n')
np.savetxt("./data.txt",data,fmt='%d',delimiter=',')