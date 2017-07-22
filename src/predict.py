#coding=utf8
from collections import Counter
import sys
import numpy as np
import getdata
import codecs,os
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def vote(votelist):
  '''
  投票
  votelist:结果的投票,取最多者,list
  '''
  return Counter(votelist).most_common(1)[0][0]


def main():
  X,indexdict = getdata.get_test('/home/lihang/2017/bdimg/data/test1/image/')
  #indexdict是图片序号和名称对应,这里图片名称还是包含了后缀名.jpg
  X = preprocess_input(X)
  #path = '/home/lihang/2017/bdimg/src76/inceptionv3-ft.model'
  path = '/home/lihang/2017/bdimg/src/2inception.model20'
  model1 = load_model(path)
  preds = model1.predict(X)
  file = codecs.open('res.txt','w','utf8')
  #将输出的标签对应成真的标签
  labelmap = getdata.makelabel2label('/home/lihang/2017/bdimg/data/train_data2/')
  rlabelmap = {}
  for i in labelmap:
        rlabelmap[labelmap[i]] = i
  for index,i in enumerate(preds):
    label = np.argmax(i)
    label = rlabelmap[label]
    file.write(str(label)+'\t'+indexdict[index][0:-4]+'\n')
  file.close()

if __name__=="__main__":
  main()