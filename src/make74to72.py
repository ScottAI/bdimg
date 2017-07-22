#coding=utf8
import codecs
inputfile = 'res1.txt'
outputfile = 'res1_1.txt'
two72 = True#实测将标签全部置为72更好一些结果
file = codecs.open(inputfile,'r','utf8')
file = file.readlines()
filelist = [i.split() for i in file]
out = codecs.open(outputfile,'w','utf8')
out.truncate()
if two72:
    for i in filelist:
        if i[0]=='74':
            out.write('72'+'\t'+i[1]+'\n')
        #elif i[0]=='132':
        #    out.write('26'+'\t'+i[1]+'\n')
        else:
            out.write(i[0]+'\t'+i[1]+'\n')
else:
    for i in filelist:
        if i[0]=='72':
            out.write('74'+'\t'+i[1]+'\n')
        else:
            out.write(i[0]+'\t'+i[1]+'\n')