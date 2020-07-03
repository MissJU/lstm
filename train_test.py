from motion_classify import models
from motion_classify import data_process
import tensorflow as tf
import warnings
import numpy as np
import os
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings('ignore')
datas=np.array(data_process.data)
labels=np.array(data_process.labels)
batch_size=64
#训练集，测试集划分
#hidden_size=128
num_layers=2
vocab_size=data_process.vocab_size
ema_rate=0.99 #移动平均衰减率
lr_decay_step=1000 #衰减频率
lr_decay=0.99 #衰减频率
lr=0.0001 #初始学习率
epoches=2000
show_step=10
save_step=100
model_name='model'
vocab_size = 100000
max_seq_num =max(data_process.review_lens)  # 句子最大长度
num_dimensions = 500  # 词向量长度
batch_size = 64  # batch的尺寸
num_classes = 2  # 输出的类别数
iterations = 1000  # 迭代的次数
lstmUnits=64
wordsList=data_process.word_list
wordVectors=data_process.reviews_ints
rnn_keep_prob=0.5
emb_keep_prob=0.5
show_step=10
save_step=100
embedding_size=300
num_layers=2
# 数据
x = tf.placeholder(tf.int32,[batch_size,max_seq_num])
# 标签
y = tf.placeholder(tf.float32, [batch_size,1])
# emb层的dropout保留率
emb_keep = tf.placeholder(tf.float32)
# rnn层的dropout保留率
rnn_keep = tf.placeholder(tf.float32)

# 建立embeddings矩阵
embeddings = tf.get_variable("embeddings", [vocab_size, embedding_size], initializer=tf.truncated_normal_initializer)

#使用dropout的LSTM
lstm_cell=[tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(lstmUnits),output_keep_prob=emb_keep) for _ in range(num_layers)]

#构建循环神经网络
cell=tf.nn.rnn_cell.MultiRNNCell(lstm_cell)

#生成词嵌入矩阵，并进行dropout
# 将词索引号转换为词向量[None, max_document_length] => [None, max_document_length, embedding_size]
embedded = tf.nn.embedding_lookup(embeddings, x)
dropout_input=tf.nn.dropout(embedded,rnn_keep)

#计算rnn的输出
outputs,last_state=tf.nn.dynamic_rnn(cell,dropout_input,dtype=tf.float32)
#二分类问题
last_outputs=outputs[:,-1,:]

# 求最后节点输出的线性加权和
weights = tf.Variable( tf.random.truncated_normal([lstmUnits,1]), dtype=tf.float32, name='weights')
bias = tf.Variable(0,dtype=tf.float32,name='bias')

predict= tf.matmul(last_outputs, weights) + bias
output = tf.nn.sigmoid(predict)
print(predict.shape)
# 真负类
ok0 = tf.logical_and(tf.less_equal(output,0.5), tf.equal(y, 0))
# 真正类
ok1 = tf.logical_and(tf.greater(output, 0.5), tf.equal(y, 1))
# 一个数组，所有预测正确的都为True,否则False
ok = tf.logical_or(ok0, ok1)
# 先转化成浮点型，再通过求平均来计算acc
acc = tf.reduce_mean(tf.cast(ok, dtype=tf.float32))
#定义损失
loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=predict)
loss=tf.reduce_mean(loss)

# #优化器
#global_step=tf.Variable(0,trainable=False)
# #学习率衰减
#learn_rate=tf.train.exponential_decay(lr,global_step,lr_decay_step,lr_decay)
#反向传播优化器
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)

#移动平均操作
# ema = tf.train.ExponentialMovingAverage(ema_rate,global_step)
# avg_op=ema.apply(tf.trainable_variables())
# model=models.Model(x,y,emb_keep,rnn_keep)

sess = tf.compat.v1.InteractiveSession()
saver = tf.compat.v1.train.Saver()
#组合构成训练op
# with tf.control_dependencies([optimizer,avg_op]):
#     train_op=tf.no_op('train')
#     #return train_op

def getTrainBatch():
    label=np.zeros([batch_size,1])
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = random.randint(1,11499)
        else:
            num = random.randint(13499,24999)
        arr[i] = datas[num-1:num]
        label[i]=labels[num-1:num]
    return arr, label



# 全局初始化
sess.run(tf.compat.v1.global_variables_initializer())
# 迭代训练
for step in range(iterations):
    # 获取一个batch进行训练
    batch_x,batch_y =getTrainBatch() #dataset.next_batch(datas,batch_size)
    #print(batch_x.shape,batch_y.shape)
    feed_dict = {x:batch_x,
                 y:batch_y,
                 emb_keep:[[emb_keep_prob]],
                 rnn_keep:[[rnn_keep_prob]]
                 }
    print(1)
    loss_,_= sess.run([loss,optimizer],feed_dict=feed_dict)
    # 输出loss
    if step % show_step == 0:
        print('step {},loss is {}'.format(step, loss_))
    # 保存模型
    if step % save_step == 0:
        saver.save(sess, os.path.join('ckpt','model'))
