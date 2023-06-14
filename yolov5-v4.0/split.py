import random
import numpy as np

getTrain = []
getTest = []
getVal = []

arrTrain = []
arrTest = []
arrVal = []

arr = np.arange(1, 1186)
# 训练集
num = 0
for i in np.random.permutation(arr):
    if num > 829:
        break
    arrTrain.append(i)
    num = num + 1
getTrain = np.array(arrTrain)
getTrain.sort()

f = open('train.txt', 'w')
for i in getTrain:
    f.write('{:04d}\n'.format(i))
f.close()
# 测试集
num = 0
for i in np.random.permutation(arr):
    if num > 178:
        break
    arrTest.append(i)
    num = num + 1
getTest = np.array(arrTest)
getTest.sort()

f = open('test.txt', 'w')
for i in getTest:
    f.write('{:04d}\n'.format(i))
f.close()
# 验证集
num = 0
for i in np.random.permutation(arr):
    if num > 178:
        break
    arrVal.append(i)
    num = num + 1
getVal = np.array(arrVal)
getVal.sort()

f = open('val.txt', 'w')
for i in getVal:
    f.write('{:04d}\n'.format(i))
f.close()
