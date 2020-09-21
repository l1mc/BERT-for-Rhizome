# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:36:47 2020

@author: Mingcong Li
"""

# 一、导入包
import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics import f1_score


# 检测计算机是否支持用GPU计算。
tf.test.is_gpu_available()
# 设置成用GPU计算。在Kaggle上实测，用GPU比用CPU快至少20倍。用CPU跑完fit大概要30min。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 下载tokenization这个包，在终端上执行。
!wget https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tokenization


# 二、定义encode和建立模型的function
# 1. 用来编码的function
def bert_encode(texts, tokenizer, max_len=512):  # 输入的内容是 文本变量 和 tokenizer
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)  # 对text分词，tokenize

        text = text[:max_len-2]  # 对text进行截取，去第0位到最大位数限制减去2位，也就是留出来两位放CLS和SEP。
        input_sequence = ["[CLS]"] + text + ["[SEP]"]  # 制作添加好CLS和SEP的句子
        pad_len = max_len - len(input_sequence)  # 最大长度减去有文字的长度，剩下的是需要pad的长度

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        # 把分好词的句子中的每一个词转换成ID。  ***不知道这个是干什么的***
        tokens += [0] * pad_len
        # 上面表示tokens里面不到max_len的条目用0填充到max_len的长度。
        # 两个list并成一个list是用加法。一个list*n，表示n个同样的list合并成一个list。
        pad_masks = [1] * len(input_sequence) + [0] * pad_len  # 有内容的部分为1，padding的部分为0
        segment_ids = [0] * max_len

        all_tokens.append(tokens)  # text转成ID后的结果。每处理完texts中的一行，就往最终结果里面增加一行。
        all_masks.append(pad_masks)  # 表征pad信息的masks
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

# 用来建立模型的function
def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    # 这里的Input()是tensorflow.keras.layers里面的Input。表示的是模型的输入层。
    # 第一个参数shape是一个表示形状的元组。
    # dtype表示输入的数据类型。
    # name表示输入层的名字。如果不写，那么系统会自动生成一个名字。
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    # 这里的Dense是tensorflow.keras.layers里面的Dense。是一个全连接层。
    # 第一个参数是一个正整数，表示该层的神经单元结点数。。
    # 第二个参数是激活函数（激励函数），即神经网络的非线性变化

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    # Model()把layers聚合成为一个对象，这个对象拥有训练和推断的能力
    # 第一个参数是输入，是一个keras.Input对象，或者是一个由keras.Input对象组成的列表。
    # 第二个参数是输出。


    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
    # 这里是配置config用于训练的模型。Adam是优化器optimizer。
    # loss是目标函数，模型求解的过程就是最小化目标函数loss的过程。
    # metrics是模型在训练以及测试的时候，进行评估的指标。

    return model


# 3. 从TensorFlow Hub上导入Bert
# 准备好SavedModel
# module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
module_url = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/2"
bert_layer = hub.KerasLayer(module_url, trainable=True)
# hub.KerasLayer()是把一个提前保存好的模型（SavedModel）打包起来，作为一个Keras层。


# 4. 导入数据并划分训练集、测试集
data = pd.read_csv("../input/xiecheng/ChnSentiCorp_htl_all.csv")
# data.isnull().any()  # 这一行以及下一行是检测NaN值的
# data[data.isnull().values==True]
data = data.drop(6374)  # 剔除Nan值的行，在这里只有一行
X_train, X_test, y_train, y_test = train_test_split(data.review.values, data.label.values, test_size=0.3, random_state=42)
unique, counts = np.unique(y_train, return_counts=True)  # 检查一下分的是否均匀
dict(zip(unique, counts))
data.isnull().any()


# 5. 生成Tokenizer
# 注意，输入的ndarray里面，不能有任何元素是NaN，否则会报错。
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# vocab_file是词典，建立了词到id的映射关系
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# BERT里分词主要是由FullTokenizer类来实现的。
# Fulltokenizer()需要传入两个参数。
# 第一个参数vocab_file是词典。
# 第二是参数do_lower_case决定了我们的模型是否区分大小写。可以传入"True"和"False"。如果我们只是Fine-Tuning，那么这个参数需要与模型一致，比如模型是uncased_L-12_H-768_A-12，那么do_lower_case就必须为True。
# Fulltokenizer里面实际有两种Tokenizer发挥了工作。
# 第一种是BasicTokenizer，根据空格等进行普通的分词。它首先被调用。
# 第二种是WordpieceTokenizer，把前者的结果再细粒度的切分为WordPiece。对于中文来说，WordpieceTokenizer什么也不干，因为之前的分词已经是基于字符的了


# 6. 把文本encode成tokens、masks、segment flag
train_input = bert_encode(X_train, tokenizer, max_len=160)
# 上面一行调用4.2里面的第一个function。
# 第一个参数是一个ndarray，类似于列表格式，是所有的文本。
# 第二个参数是4.5里面定义的tokenizer
# 这里是把文本进行encoding，生成三个结果：tokens，masks，segment flags。下面一行同理。
test_input = bert_encode(X_test, tokenizer, max_len=160)
train_labels = y_train


# 7. 建立并训练模型，预测
# 建立
model = build_model(bert_layer, max_len=160)  # bert_layer是在4.3里面导入的SavedModel
model.summary()
# 训练
train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    batch_size=16
)
# 保存
model.save('model.h5')  # 把这个模型保存起来，命名为‘model.h5’
# 预测
test_pred = model.predict(test_input)


# 8. 预测结果评估
test_pred=test_pred>0.5
test_pred=test_pred+0
type(test_pred)
f1 = f1_score(y_test, test_pred, average='binary')  # f1 score
print(f1)



