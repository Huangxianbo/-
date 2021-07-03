#!/usr/bin/env python
# coding: utf-8

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# ## 环境配置   本代码主要参考官方baseline
# 
# 目前飞桨（PaddlePaddle）正式版仍为1.8.4，以下代码均为2.0RC0测试版本，本地安装需要指定版本号进安装。

# ## 数据预处理 - 配置部分 （代码并没有保存CheckPoint）

# In[2]:


import os
import pandas as pd
import numpy as np
from paddle.io import Dataset
from baseline_tools import *

DATA_RATIO = 0.9  # 训练集和验证集比例

# None表示不使用，“emb”为Embedding预处理方案，选手可自由选择使用字段以及预处理方案
TAGS = {'android_id': None,
        'apptype': "emb",
        'carrier': "emb",
        'dev_height': 'emb',
        'dev_ppi': 'emb',
        'dev_width': 'emb',
        'lan': 'emb',
        'media_id': "emb",
        'ntt': "emb",
        'os': "emb",
        'osv': 'emb',
        'package': "emb",
        'sid': None,
        'timestamp': "norm",
        'version': "emb",
        'fea_hash': 'norm',
        'location': "emb",
        'fea1_hash': 'norm',
        'cus_type': 'emb'}

# 归一化权重设置


# ## 数据预处理 - 生成Embedding所需数据

# In[3]:


datas = pd.read_csv("train.csv")
for ids,data in enumerate(datas["fea_hash"]):
    try:
        data = float(data)
    except:
        datas["fea_hash"][ids] = 499997879
        print(ids+1)
datas.to_csv("train.csv")
#fea_hash字段中有许多异常数据，将他们改为最大值的一半：499997879.0
datas = pd.read_csv("test.csv",dtype=str)
#datas = datas["fea_hash"]
#print(datas.head)

for ids,data in enumerate(datas["fea_hash"]):
    try:
        data = float(data)
    except:
        datas["fea_hash"][ids] = 499997879
        print(ids+1)
datas = datas
datas.to_csv("test.csv")
print('complete')


# In[4]:


TRAIN_PATH = "train.csv"
SAVE_PATH = "emb_dicts"
df = pd.read_csv(TRAIN_PATH, index_col=0)
pack = dict()
for tag, tag_method in TAGS.items():
    if tag_method != "emb":
        if tag_method == 'norm':
            print('{}字段的最小值为{}，normal字段{}'.format(tag,float(min(df.loc[:,tag])),1/(float(max(df.loc[:,tag]))-float(min(df.loc[:,tag])))))
        continue
    else:
        data = df.loc[:, tag]
        dict_size = make_dict_file(data, SAVE_PATH, dict_name=tag)
        pack[tag] = dict_size + 1  # +1是为了增加字典中不存在的情况，提供一个默认值

with open(os.path.join(SAVE_PATH, "size.dict"), "w", encoding="utf-8") as f:
    f.write(str(pack))

print("全部生成完毕")


# ## 数据预处理 - 定义数据读取器以及预处理流程

# In[5]:


def get_size_dict(dict_path="./emb_dicts/size.dict"):
    """
    获取Embedding推荐大小
    :param dict_path: 由run_make_emb_dict.py生成的size.dict
    :return: 推荐大小字典{key: num}
    """
    with open(dict_path, "r", encoding="utf-8") as f:
        try:
            size_dict = eval(f.read())
        except Exception as e:
            print("size_dict打开失败，请检查", dict_path, "文件是否正常，报错信息如下:\n", e)
        return size_dict
class Data2IdNorm:
    """
    数据归一化类
    """
    def __init__(self, norm_weight,norm_min):
        self.norm_weight = norm_weight
        self.norm_min = norm_min
    def transform_data(self, sample, shape=None, d_type="float32"):
        sample = (float(sample)-self.norm_min)*self.norm_weight
        sample = value2numpy(sample, shape, d_type)
        return sample

    def get_method(self):
        return self.transform_data

class Reader(Dataset):
    def __init__(self,
                 is_infer: bool = False,
                 is_val: bool = False,
                 use_mini_train: bool = False,
                 emb_dict_path="./emb_dicts"):

        """
        数据读取类
        :param is_infer: 是否为预测Reader
        :param is_val: 是否为验证Reader
        :param use_mini_train：使用Mini数据集
        :param emb_dict_path: emb字典路径
        """
        super().__init__()
        # 选择文件名
        train_name = "mini_train" if use_mini_train else "train"
        file_name = "test" if is_infer else train_name
        # 根据文件名读取对应csv文件
        df = pd.read_csv(file_name + ".csv")
        # 划分数据集
        if is_infer:
            self.df = df.reset_index(drop=True)
        else:
            start_index = 0 if not is_val else int(len(df) * DATA_RATIO)
            end_index = int(len(df) * DATA_RATIO) if not is_val else len(df)
            self.df = df.loc[start_index:end_index].reset_index(drop=True)
        # 数据预处理
        NORM_WEIGHT = {'timestamp':1.6534305617481573e-09,'fea_hash':2.3283201561138293e-10,'fea1_hash':2.3299594677571534e-10}
        zuixiaozhi = {'timestamp':1559491201174.7812,'fea_hash':0.0,'fea1_hash':12400.0}
        self.cols = [tag for tag, tag_method in TAGS.items() if tag_method is not None]
        self.methods = dict()
        for col in self.cols:
            # ===== 预处理方法注册 =====

            if TAGS[col] == "emb":
                self.methods[col] = Data2IdEmb(dict_path=emb_dict_path, dict_name=col).get_method()
            elif TAGS[col] == "norm":
                self.methods[col] = Data2IdNorm(norm_weight=NORM_WEIGHT[col],norm_min=zuixiaozhi[col]).get_method()

        # 设置FLAG负责控制__getitem__的pack是否包含label
        self.add_label = not is_infer
        # 设置FLAG负责控制数据集划分情况
        self.is_val = is_val

    def __getitem__(self, index):
        """
        获取sample
        :param index: sample_id
        :return: sample
        """
        # 因为本次数据集的字段非常多，这里就使用一个列表来"收纳"这些数据
        pack = []
        # 遍历指定数量的字段
        for col in self.cols:
            sample = self.df.loc[index, col]
            sample = self.methods[col](sample)
            pack.append(sample)

        # 如果不是预测，则添加标签数据
        if self.add_label:
            tag_data = self.df.loc[index, "label"]
            tag_data = np.array(tag_data).astype("int64")
            pack.append(tag_data)
            return pack
        else:
            return pack

    def __len__(self):
        return len(self.df)
        


# ## 数据预处理 - 检查数据是否可以正常读取（可选）
# 默认只检查训练，infer和test可以在`val_reader = Reader(此处设置)`中参考刚刚定义的`Reader`来配置

# In[ ]:


# 用于独立测试数据读取是否正常 推荐在本地IDE中下断点进行测试
print("检查数据ing...")
val_reader = Reader()

print(len(val_reader))
for data_id, data in enumerate(val_reader):
  
    for i in range(len(data)):
        if data_id == 0:
            print("第", i, "个字段 值为：", data[i])
        else:
            break
    if data_id % 1000 == 0:
        print("第", data_id, "条数据可正常读取 正在检查中", end="\r")
    if data_id == len(val_reader) - 1: 
        print("数据检查完毕")
        break


# In[ ]:


print(val_reader.methods)


# **以上对于数据的预处理参考了官方的数据处理方法，自己尝试了几种其他的组合方案但是效果并不好。所以最后还是采用了官方的baseline方案，这里应该还有不少的优化空间。**

# ## 训练&推理配置

# In[4]:


import os
import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.tensor as tensor
from paddle.static import InputSpec
from paddle.metric import Accuracy

# 模型保存与加载文件夹
SAVE_DIR = "./output/"

# 部分训练超参数
EPOCHS = 30  # 训练多少个循环
TRAIN_BATCH_SIZE = 128  # mini_batch 大小
EMB_SIZE = 64  # Embedding特征大小
EMB_LINEAR_SIZE =  128 # Embedding后接Linear层神经元数量
LINEAR_LAYERS_NUM = 8  # 归一化方案的Linear层数量
class line(nn.Layer):
    def __init__(self,dim,hidden_dim):
        super().__init__()
        self.lay = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim,dim)
        )
    def forward(self,x):
        return self.lay(x)
class mat(nn.Layer):
    def __init__(self,dim,heads):
        super().__init__()
        dim_head = dim//heads
        self.heads = heads
        inner_dim = dim * heads
        self.dh = dim_head**-0.5
        self.to_output = nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(0.2)
        )
        self.to_kqv = nn.Linear(dim,inner_dim*3,bias_attr=None)
    def forward(self,x):
        b,n,_ = x.shape
        qkv = self.to_kqv(x).chunk(3,axis=-1)
        q,k,v = map(lambda t: paddle.reshape(t,[b,self.heads,n,-1]),qkv)
        dots = paddle.matmul(q,k,transpose_y=True)*self.dh
        soft = nn.functional.softmax(dots,axis= -1)
        out =paddle.matmul(soft,v)
        out = paddle.reshape(out,[b,n,-1])
        out = self.to_output(out)
        return out
class PreNorm(nn.Layer):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self,x):
        return self.fn(self.norm(x))
class transformer(nn.Layer):
    def __init__(self,dim,number):
        super().__init__()
        self.lay = nn.LayerList([])
        for _ in range(number):
            self.lay.append(
            nn.LayerList([cancha(PreNorm(dim,mat(dim,8))),
            cancha(PreNorm(dim,line(dim,int(dim/2))))
            ]))
            
    def forward(self,x):
        for lay1,lay2 in self.lay:
            x = lay1(x)
            x = lay2(x)
        return x
# 配置训练环境
USE_MINI_DATA = False  # 默认使用小数据集，此方法可加快模型产出速度，但可能会影响准确率
class cancha(nn.Layer):
    def __init__(self,fc):
        super().__init__()
        self.fc = fc
  
    def forward(self,x):
        y = self.fc(x)

        y += x
        return y
# 组网
class SampleNet(paddle.nn.Layer):
    def __init__(self, tag_dict: dict, size_dict: dict):
        # 继承Model
        super().__init__()
        # 新建一个隐藏层列表，用于存储各字段隐藏层对象
      
        # 定义一个用于记录输出层的输入大小变量，经过一个emb的网络结构就增加该结构的output_dim，以此类推
        out_layer_input_size = 0
        # 遍历每个字段以及其处理方式
        self.hidden_layers_list = nn.LayerList([])
        for tag, tag_method in tag_dict.items():
            # ===== 网络结构方法注册 =====
            # Embedding方法注册
            if tag_method == "emb":
                emb = nn.Embedding(num_embeddings=size_dict[tag], embedding_dim=EMB_SIZE)#定义多个字段的emb处理方法，添加到一个列表中
                self.hidden_layers_list.append(emb)
            elif tag_method == "norm":
                continue
            elif tag_method is None:
                continue
        self.dict_list = ['emb','emb','emb','emb','emb', 'emb', 'emb', 'emb', 'emb','emb', 'norm', 'emb', 'norm', 'emb', 'norm', 'emb']
     
        self.con1= nn.Conv1D(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding = 1)

        
        #self.hidden_layer = nn.LSTM(EMB_SIZE, 128, 1)
        #self.hidden_layer = mat(EMB_SIZE,8)
        self.hidden_layer = transformer(EMB_SIZE,5)
        '''
        self.hidden_layer1 = nn.Sequential(
                                        nn.Conv1D(in_channels=1,out_channels=32,kernel_size=8,stride=8),
                                        nn.ReLU(),
                                        nn.Conv1D(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.Conv1D(in_channels=16,out_channels=1,kernel_size=3,stride=1,padding=1)
                                            
                                        )
        '''
        self.con2 = nn.Conv1D(in_channels=32,out_channels=1,kernel_size=3,stride=1,padding = 1)
        self.hidden_layer2=nn.Sequential(  
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=16,out_features=8)
                                                )
        
        
        # 归一化方法注册
    
        self.hidden_layer_norm1 =  nn.LSTM(1,32)
        self.hidden_layer_norm2 = transformer(32,5)
        self.hidden_layer_norm=nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(in_features= 32,out_features = 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features = 16,out_features = 1)
        )
        # 定义输出层，因为是二分类任务，激活函数以及损失方案可以由选手自己发挥，此处为sigmoid激活函数
        # Tips: 若使用sigmoid激活函数，需要修改output_dim和损失函数，推荐理解原理后再尝试修改
        self.out_layers = nn.Sequential(
            nn.LayerNorm(107),
            nn.Linear(in_features=107,out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64,out_features=2)
            )

    # 前向推理部分 `*input_data`的`*`表示传入任一数量的变量
    def forward(self, *input_data):
        layer_list = []  # 用于存储各字段特征结果
        number = 0
        for sample_data ,tag_method in zip(input_data,self.dict_list):
            
            if tag_method == 'norm':
                tmp = sample_data
                tmp = tmp.astype('float32')
                tmp = tmp.unsqueeze(1)
                tmp,(l,m) = self.hidden_layer_norm1(tmp)
                tmp = self.hidden_layer_norm2(tmp)
                tmp = self.hidden_layer_norm(tmp)
            else:
                emb = self.hidden_layers_list[number]
                tmp = emb(sample_data.astype('int64'))
                tmp = tmp.astype('float32')
                #tmp = self.con1(tmp)
                #tmp = self.hidden_layer(tmp)
                #tmp = self.con2(tmp)
                #tmp = self.hidden_layer1(tmp)
                tmp = self.hidden_layer(tmp)
                tmp = self.hidden_layer2(tmp)
                number += 1
            layer_list.append(tensor.flatten(tmp, start_axis=1))  # flatten是因为原始shape为[batch size, 1 , *n], 需要变换为[bs, n]
        # 对所有字段的特征合并
        
        layers = tensor.concat(layer_list, axis=1)
        # 把特征放入用于输出层的网络
        result = self.out_layers(layers)
        #result = paddle.nn.functional.softmax(result)
        # 返回分类结果
        return result
# 定义网络输入
inputs = []
for tag_name, tag_m in TAGS.items():
    d_type = "float32"
    if tag_m == "emb":
        d_type = "int64"
    if tag_m is None:
        continue
    inputs.append(InputSpec(shape=[-1, 1], dtype=d_type, name=tag_name))


# **这里采用了最近很火的transformer框架，并且加上了LSTM进行优化，本来还加上了卷积部分，但是加上卷积之后训练时间变得过长，并且准确率并没有什么提升所以还是放弃了卷积结构**

# ### 执行训练

# In[ ]:


use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
place = paddle.CUDAPlace(0)
learn = paddle.optimizer.lr.StepDecay(learning_rate=0.007,step_size=3516,gamma=0.8,verbose = False)
# 定义Label
labels = [InputSpec([-1, 1], 'int64', name='label')]
# 实例化SampleNet
model = paddle.Model(SampleNet(TAGS, get_size_dict()), inputs=inputs, labels=labels)
# 获取训练集和测试集数据读取器
train_reader = Reader(use_mini_train=USE_MINI_DATA)
val_reader = Reader(use_mini_train=USE_MINI_DATA, is_val=True)
# 定义优化器
optimizer = paddle.optimizer.Adam(learning_rate=learn, parameters=model.parameters())
# 模型训练配置
model.prepare(optimizer, paddle.nn.loss.CrossEntropyLoss(), Accuracy())
# 开始训练
model.fit(train_data=train_reader,  # 训练集数据
            eval_data=val_reader,  # 交叉验证集数据
            batch_size=TRAIN_BATCH_SIZE,  # Batch size大小
            epochs=EPOCHS,  # 训练轮数
            log_freq=500,  # 日志打印间隔
            save_dir=SAVE_DIR)  # checkpoint保存路径


# ### 执行推理

# In[ ]:


# 推理部分
CHECK_POINT_ID = "1"  # 如没有训练完毕，可以挑选SAVE_DIR路径的下的中的checkpoint文件名(不包含拓展名哦)，例如"1"
TEST_BATCH_SIZE = 128  # 若因内存/显存发生报错，请优先调整为1
RESULT_FILE = "./result1.csv"  # 推理文件保存位置

# 实例化SampleNet
model = paddle.Model(SampleNet(TAGS, get_size_dict()), inputs=inputs)
# 获取推理Reader并读取参数进行推理
infer_reader = Reader(is_infer=True)
model.load(os.path.join(SAVE_DIR, CHECK_POINT_ID))
# 开始推理
model.prepare()
infer_output = model.predict(infer_reader, TEST_BATCH_SIZE)
# 获取原始表中的字段并添加推理结果
result_df = infer_reader.df.loc[:, "sid"]
pack = []
for batch_out in infer_output[0]:
    for sample in batch_out:
        pack.append(np.argmax(sample))
# 保存csv文件
result_df = pd.DataFrame({"sid": np.array(result_df, dtype="int64"), "label": pack})
result_df.to_csv(RESULT_FILE, index=False)
print("结果文件保存至：", RESULT_FILE)


# ## 代码审查
# 如选手成绩靠前并收到官方邮件通知代码审查，请参考该[链接](https://aistudio.baidu.com/aistudio/projectdetail/743661)进行项目上传操作  
# 快捷命令:`!zip -rP [此处添加审查邮件中的Key值] [邮件中的UID值].zip /home/aistudio/`

# In[1]:


get_ipython().system('zip -r [黄贤博].zip /home/aistudio/')

