#coding=utf8
import codecs
inputfile = 'res7.txt'
outputfile = 'res7_1.txt'
two72 = True#实测将标签全部置为72更好一些结果
file = codecs.open(inputfile,'r','utf8')
file = file.readlines()
filelist = [i.split() for i in file]
out = codecs.open(outputfile,'w','utf8')
out.truncate()
inx = 0
label = 0
for i in filelist:
    if inx%2==0:
        out.write(str(label)+'\t'+i[1]+'\n')
    else:
        out.write(i[0]+'\t'+i[1]+'\n')
    inx = inx + 1
