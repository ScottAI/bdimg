#coding=utf8
#为了简单,以vote投票形式的融合就以每个模型的输出结果而后这样程序再对比去生成最后投票结果来做吧.
import codecs
from collections import Counter

def readfile(filename):
    file1 = codecs.open(filename,'r','utf8')
    file1list = file1.readlines()
    file1list = [i.split('\t') for i in file1list]
    file1list = [[i[0],i[1].replace('\n','')] for i in file1list]
    return file1list

def vote(votelist):
    '''
    votelist:结果的投票,取最多者,list
    '''
    b = Counter(votelist)
    res = b.most_common(1)[0][0]
    restime = b[res]
    houxuan = []
    for i in b:
        if b[i]==restime:
            houxuan.append(i)
    houxuan = set(houxuan)
    for i in votelist:
        if i in houxuan:
            return i


filelist = ['res29.txt','res30.txt','res21.txt','res20.txt','res16.txt','res12.txt','res10.txt','res8.txt','res7.txt','res6.txt','res5.txt','res4.txt','res3.txt', 'res2.txt']#将最可信的放在最开始
#filelist = ['res26.txt','res25.txt','res24.txt','res23.txt','res22.txt','res21.txt','res20.txt']
#filelist = [ 'res15.txt','res14.txt','res13.txt','res12.txt','res11.txt','res10.txt','res9.txt', 'res8.txt' ]#将最可信的放在最开始
length = len(filelist)

#假定最后一行的回车已经去除,关于回车实测提交时不会有影响
#这里以第一个文件的id为基准
file1list = readfile(filelist[0])
print(len(file1list))
filedict = {i[1]:[int(i[0])] for i in file1list}#id:标签列表
print(len(filedict))

for i in xrange(length):
    if i ==0:#忽略第一个文件
        continue
    else:
        filelistnow = readfile(filelist[i])
        for j in filelistnow:
            filedict[j[1]].append(int(j[0]))
print(filedict['2467456860,3136914853'])

#进行投票
resdict = {}
for i in filedict:
    resdict[i]=vote(filedict[i])

resfile = codecs.open('merge.txt','w','utf8')
for i in file1list:
    #id:i[1]
    resfile.write(str(resdict[i[1]])+'\t'+i[1]+'\n')
resfile.close()


