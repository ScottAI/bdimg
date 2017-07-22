#coding=utf8
import codecs
import os
import numpy as np
from PIL import Image
import tensorlayer as tl
#root_path = '/home/lihang/2017/bdimg/data/train_data/'

def makelabel2label(root_path):
    li = os.listdir(root_path)
    labelmap = {}
    for index,i in enumerate(li):
        labelmap[i] = index
    return labelmap

def get_path_label(root_path):
    '''
    获取文件的文件名(完整路径)与对应标签的映射字典
    return:{文件名1:label,文件名2:label2,...}
    '''
    labelmap = makelabel2label(root_path)
    pathdict = {}
    li = os.listdir(root_path)
    for i in li:
        pathdict[i] = root_path+i+'/'
    pathlabel = {}
    for label in pathdict:#对每一个label
        filelist = os.listdir(pathdict[label])
        for everyfile in filelist:#对每一个label目录下的文件
            pathlabel[pathdict[label] + everyfile] = labelmap[label]
    return pathlabel
#print(len(get_path_label()))

def get_image(image_path,height,width):  
    """
    从给定路径中读取图片，返回的是numpy.ndarray
    image_path:string, height:图像像素高度 width:图像像素宽度
    return:numpy.ndarray的图片tensor 
    """ 
    im = Image.open(image_path)
    #im = np.array(im.resize((height,width),Image.BILINEAR))
    #b = np.reshape(im,[height,width,1])
    b = np.reshape(im, [im.size[1], im.size[0], 3])
    b = tl.prepro.imresize(b, size=(height, width), interp='bilinear')
    return b

def get(rootpath,image_height=299,image_width=299,image_channel=3,make=False):
    pathlabel = get_path_label(rootpath)
    image_num = len(pathlabel)
    inx = 0
    X = np.zeros((image_num, image_height, image_width, image_channel), np.float32)
    Y = np.zeros((image_num,1),np.uint8)
    for path in pathlabel:#对每一个label
        data = get_image(path,image_height,image_width)
        label = pathlabel[path]
        X[inx,:,:,:] = data
        Y[inx,:] = label
        inx = inx+1
        #break
    print(X.shape,Y.shape)
    return X,Y

#X, Y = get()
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1, random_state=33)
def get_test(root_path,image_height=299,image_width=299,image_channel=3):
    li = os.listdir(root_path)
    image_num = len(li)
    X = np.zeros((image_num, image_height, image_width, image_channel), np.float32)
    inx = 0
    indexdict = {}
    for i in li:
        indexdict[inx] = i
        npath = root_path+i
        data = get_image(npath,image_height,image_width)
        X[inx,:,:,:] = data
        inx = inx+1
    return X,indexdict

def get_dict(filepath):
    file = codecs.open(filepath)
    file = file.readlines()
    file = [i.split('\t') for i in file]
    res = {i[0]:i[1] for i in file}
    return res
#X = get_test('/home/lihang/2017/bdimg/data/test1/')
#print(X.shape)
