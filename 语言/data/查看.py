# coding:utf-8
"""
Author  : Tian
Time    : 2023-06-15 10:48
Desc:
"""
import pickle
import numpy as np


# Open the file in binary mode
with open('train2.sav', 'rb') as file:
    data = pickle.load(file)

# Check the structure and size of the embeddings
print(type(data[0]["emd"]))
print(data[0]["emd"].shape)