import torch
import argparse
from torch.utils.data import DataLoader
import torchvision as tv #noqa
from typing import Type
from typing import Union
from typing import Optional
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional

import os
import glob
import json
import sys
import time
import math
import signal
import argparse
import numpy as np
from collections import defaultdict
import contextlib
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn

# from model_eee import JMPF  #485
from model2_39K import JMPF #
from data_prepro_39k import data_prepro
from data_prepro_39k_test import data_prepro_t

from pytorch_lightning.loggers import TensorBoardLogger
filename = "baseline_39K2"

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"
devices = [0]

# ckp_path = '/home/et23-maixj/mxj/JMPF_ori/checkpoint/ec-l-nnn.ckpt'
# ckp_path = '/home/et23-maixj/mxj/JMPF_state/checkpoint/baseline_mafw_2_7.ckpt'
ckp_path = None


#集成微调  

#Q2 e #Q乘矩阵 一个L14，全点乘Q 一个B32，无变化
#Q3 eeeeeQ乘矩阵 一个L14，全点乘Q 一个B32，最后一层换自注意且可训练

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
USE_CUDA = torch.cuda.is_available()
device_ids_parallel = [0,1,2]
device = torch.device("cuda:{}".format(device_ids_parallel[0]) if USE_CUDA else "cpu")
#4 b 5 l 6
# train_ratio, valid_ratio, test_ratio = 0.85, 0.15, 0.0
BATCH_TRAIN = 32
BATCH_TEST = 32
WORKERS_TRAIN = 8
WORKERS_TEST = 8
EPOCHS = 40
LOG_INTERVAL = 50
SAVED_MODELS_PATH = os.path.join(os.path.expanduser('~'), 'saved_models')
# 设置矩阵乘法的精度为中等或高等
torch.set_float32_matmul_precision('medium')
# check_p = torch.load('/home/et23-maixj/mxj/JMPF_ori/checkpoint/ec-l-v1.ckpt', map_location = device)
# '/home/et23-maixj/mxj/JMPF_ori/checkpoint/ec-l.ckpt'
model = JMPF()

# checkpoint = torch.load(ckp_path, map_location = 'cpu') # 这里假设您的checkpoint文件名是best-checkpoint.ckpt，您可以根据您的实际文件名来修改
# ckp_path = None
# model.load_state_dict(checkpoint["state_dict"],strict=False)

del_key = []
# for key in check_p["state_dict"].keys():
#         if 'feate' in key or 'seq_transformerClip' in key:
#                 del_key.append(key)

# for key in del_key:
#         del check_p["state_dict"][key]

# model.load_state_dict(check_p["state_dict"], strict=False)
# # Set up the data loader
dataset = data_prepro(root='/home/wangy/DFER_Datasets')
dataset_t = data_prepro_t(root='/home/wangy/DFER_Datasets')
len_dataset = dataset.__len__()
print('len_dataset:', len_dataset)

# train_len = int(train_ratio * len_dataset)  # 训练集长度
# valid_len = math.ceil(valid_ratio * len_dataset)  # 验证集长度
# test_len = len_dataset - train_len - valid_len  # 测试集长度

# train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[
#                                                     train_len, valid_len, test_len], generator=torch.Generator().manual_seed(0))

# train_dataset, val_dataset = random_split(dataset, lengths=[
#                                                     train_len, valid_len], generator=torch.Generator().manual_seed(0))


batch_train_size = BATCH_TRAIN  # 定义批次大小
batch_test_size = BATCH_TEST  # 定义批次大小
shuffle = True  # 定义是否打乱数据
# train_loader = DataLoader(
# train_dataset, batch_size=batch_train_size, shuffle=shuffle, num_workers=2)
# val_loader = DataLoader(
# val_dataset, batch_size=batch_train_size, shuffle=False, num_workers=2)
# test_loader = DataLoader(
# test_dataset, batch_size=batch_test_size, shuffle=False, num_workers=2)
# 创建DataLoader实例，传入dataset实例和其他参数

train_loader = DataLoader(
dataset, batch_size=batch_train_size, shuffle=shuffle, num_workers=2)
val_loader = DataLoader(
dataset_t, batch_size=batch_train_size, shuffle=False, num_workers=2)

logger = TensorBoardLogger('/home/et23-maixj/mxj/JMPF_state/runs', name='JMPF')

checkpoint_callback = ModelCheckpoint(
dirpath="/home/et23-maixj/mxj/JMPF_state/checkpoint",
filename=filename,
save_top_k=1,  # 保存最好的1个checkpoint
verbose=True,
monitor="val_loss",  # 根据验证集损失来判断最好的checkpoint
mode="min"  # 最小化验证集损失
)

trainer = Trainer(
max_epochs=EPOCHS,
accelerator="gpu",
devices=devices,
# accelerator="cpu",
callbacks=[checkpoint_callback],  # 是否启用checkpoint回调
strategy='ddp_find_unused_parameters_true',
logger=logger
)

trainer.fit(model, train_dataloaders=train_loader,
        val_dataloaders=val_loader, ckpt_path=ckp_path)

trainer.test(model,val_loader)



