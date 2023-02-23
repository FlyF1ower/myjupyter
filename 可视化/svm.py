#!/usr/bin/env python
# coding: utf-8

# ## 将txt转换为xlsx

# In[2]:


import pandas

# fdir = "pos.txt"
# my_data_dict = {}
# f = open(fdir, "r", encoding="utf-8")
# lines = f.readlines()
#
# for line in lines:
#     line = line.split()
#     str = ""
#     for word in line:
#         if word.__eq__(line[0]):  # 去掉首个单词的空格
#             str = word
#         else:
#             str += ' ' + word
#     my_data_dict[str] = 1
#
# f.close()
#
# my_excel = pandas.DataFrame({"comment": list(my_data_dict.keys())})
# my_excel.to_excel("pos.xlsx", index=False)
#
# print("done!")


# In[ ]:


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import jieba as jb
import joblib
from sklearn.svm import SVC
from gensim.models.word2vec import Word2Vec


# In[ ]:


neg =pd.read_excel("data/neg.xls",header=None, index_col=None)
pos =pd.read_excel("data/pos.xls",header=None, index_col=None)
# 这是两类数据都是x值
pos['words'] = pos[0].apply(lambda x:list(jb.cut(x)))
neg['words'] = neg[0].apply(lambda x:list(jb.cut(x)))
#需要y值  0 代表neg 1代表是pos
y = np.concatenate((np.ones(len(pos)),np.zeros(len(neg))))
X = np.concatenate((pos['words'],neg['words']))


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3)
#保存数据
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)
print("done")


# ## 对句子中的所有词向量取均值，来生成一个句子的vec

# In[3]:


def build_vector(text,size,wv):
    #创建一个指定大小的数据空间
    vec = np.zeros(size).reshape((1,size))
    #count是统计有多少词向量
    count = 0
    #循环所有的词向量进行求和
    for w in text:
        try:
            vec +=  wv[w].reshape((1,size))
            count +=1
        except:
            continue
        
    #循环完成后求均值
    if count!=0:
        vec/=count
    return vec


# In[ ]:


#初始化模型和词表
wv = Word2Vec(vector_size=300,min_count=10)
wv.build_vocab(X_train)
# 训练并建模
wv.train(X_train, total_examples=1, epochs=1)
#获取train_vecs
train_vecs = np.concatenate([build_vector(z,300,wv) for z in X_train])
#保存处理后的词向量
np.save('data/train_vecs.npy',train_vecs)
#保存模型
wv.save("data/model3.pkl")

wv.train(X_test,total_examples=1, epochs=1)
test_vecs = np.concatenate([build_vector(z,300,wv) for z in X_test])
np.save('data/test_vecs.npy',test_vecs)


# ## 训练SVM模型

# In[ ]:


#创建SVC模型
cls = SVC(kernel="rbf",verbose=True)
#训练模型
cls.fit(train_vecs,y_train)
#保存模型
joblib.dump(cls,"data/svcmodel.pkl")
#输出评分
print(cls.score(test_vecs,y_test))


# In[ ]:


from tkinter import *
import numpy as np
import jieba as jb
import joblib
from gensim.models.word2vec import Word2Vec

class core():
    def __init__(self,str):
        self.string=str

    def build_vector(self,text,size,wv):
        #创建一个指定大小的数据空间
        vec = np.zeros(size).reshape((1,size))
        #count是统计有多少词向量
        count = 0
        #循环所有的词向量进行求和
        for w in text:
            try:
                vec +=  wv[w].reshape((1,size))
                count +=1
            except:
                continue
        #循环完成后求均值
        if count!=0:
            vec/=count
        return vec
    def get_predict_vecs(self,words):
        # 加载模型
        wv = Word2Vec.load("data/model3.pkl")
        #将新的词转换为向量
        train_vecs = self.build_vector(words,300,wv)
        return train_vecs
    def svm_predict(self,string):
        # 对语句进行分词
        words = jb.cut(string)
        # 将分词结果转换为词向量
        word_vecs = self.get_predict_vecs(words)
        #加载模型
        cls = joblib.load("data/svcmodel.pkl")
        #预测得到结果
        result = cls.predict(word_vecs)
        #输出结果
        if result[0]==1:
            return "好感"
        else:
            return "反感"
    def main(self):
        s=self.svm_predict(self.string)
        return s

root=Tk()
root.title("情感分析")
sw = root.winfo_screenwidth()
#得到屏幕宽度
sh = root.winfo_screenheight()
#得到屏幕高度
ww = 500
wh = 300
x = (sw-ww) / 2
y = (sh-wh) / 2-50
root.geometry("%dx%d+%d+%d" %(ww,wh,x,y))
# root.iconbitmap('tb.ico')

lb2=Label(root,text="输入内容，按回车键分析")
lb2.place(relx=0, rely=0.05)

txt = Text(root,font=("宋体",20))
txt.place(rely=0.7, relheight=0.3,relwidth=1)

inp1 = Text(root, height=15, width=65,font=("宋体",18))
inp1.place(relx=0, rely=0.2, relwidth=1, relheight=0.4)

def run1():
    txt.delete("0.0",END)
    a = inp1.get('0.0',(END))
    p=core(a)
    s=p.main()
    print(s)
    txt.insert(END, s)   # 追加显示运算结果

def button1(event):
    btn1 = Button(root, text='分析', font=("",12),command=run1) #鼠标响应
    btn1.place(relx=0.35, rely=0.6, relwidth=0.15, relheight=0.1)
    # inp1.bind("<Return>",run2) #键盘响应

button1(1)
root.mainloop()

