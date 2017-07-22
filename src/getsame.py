#coding=utf8
import os
target_file = '/home/lihang/2017/bdimg/data/test1/image/'
target_list = os.listdir(target_file)
print(len(target_list))

def makelabel2label(root_path):
    li = os.listdir(root_path)
    labelmap = {}
    for index,i in enumerate(li):
        labelmap[i] = i
    return labelmap
def get_path_label(root_path='/home/lihang/2017/bdimg/data/train_data2/'):
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
            pathlabel[everyfile] = labelmap[label]
    return pathlabel
source_dict = get_path_label()

read_dict = {}
for i in target_list:
    if i in source_dict:
        read_dict[i[0:-4]] = source_dict[i]
print(len(read_dict))

import codecs
answer_dict = {}
file1 = codecs.open('merge.txt','r','utf8').readlines()
file1 = [i.replace('\n','') for i in file1]
file1 = [i.split() for i in file1]
for i in file1:
    answer_dict[i[1]]=i[0]
for i in file1:
    if i[1] in read_dict:
        if i[0] != read_dict[i[1]]:
            print('o  ',i[1])
            answer_dict[i[1]] = read_dict[i[1]]
file2 = codecs.open('merge.txt','w','utf8')
for i in answer_dict:
    file2.write(answer_dict[i]+'\t'+i+'\n')