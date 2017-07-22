#coding=utf8
'''
整理train文件夹里的数据到train_data里,方式为同一类别的放到同一文件夹下
'''
import os
import shutil
classnum = 134
root_path = '/home/lihang/2017/bdimg/data/'
def makedir():
    for i in range(classnum):
        nowpath = root_path+'train_data/'+str(i)+'/'
        if not os.path.exists(nowpath):
            os.mkdir(root_path+str(i))

import codecs
with codecs.open(root_path+'val.txt') as labelfile:
    labelfile = labelfile.readlines()
    labellist = [i.split()[0:2] for i in labelfile]
    labelmap = {i[0]+'.jpg':i[1] for i in labellist}

for i in os.listdir(root_path+'train/'):
    if i in labelmap:
        shutil.copy(root_path+'train/'+i,root_path+'train_data/'+str(labelmap[i])+'/'+i)