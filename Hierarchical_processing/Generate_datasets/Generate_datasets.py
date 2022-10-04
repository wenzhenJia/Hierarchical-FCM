# -*- coding: utf-8 -*-
#在程序目录下创建output目录
# 读取csv文件
import csv
import time
import h5py
import numpy as np
cvspathlist=['D://Hierarchical-FCM-ResNets//Hierarchical_processing//Generate_datasets//NYCdata//2014-04 - Citi Bike trip data.csv',
             'D://Hierarchical-FCM-ResNets//Hierarchical_processing//Generate_datasets//NYCdata//2014-05 - Citi Bike trip data.csv',
             'D://Hierarchical-FCM-ResNets//Hierarchical_processing//Generate_datasets//NYCdata//2014-06 - Citi Bike trip data.csv',
             'D://Hierarchical-FCM-ResNets//Hierarchical_processing//Generate_datasets//NYCdata//2014-07 - Citi Bike trip data.csv',
             'D://Hierarchical-FCM-ResNets//Hierarchical_processing//Generate_datasets//NYCdata//2014-08 - Citi Bike trip data.csv',
             'D://Hierarchical-FCM-ResNets//Hierarchical_processing//Generate_datasets//NYCdata//2014-09 - Citi Bike trip data.csv']#填写需要处理的文件个数,z注意路径斜杠数目
clusterpath='D://Hierarchical-FCM-ResNets//Hierarchical_processing//Generate_datasets//cluster_6.txt'

#读取类别文件
def readcluster(path):
    file = open(path, 'rb')
    line_list=[]
    while True:
        line = file.readline()
        if not line :
            break
        else:
            list = line.split("\t")
            list[2] = list[2].replace('\n', '')
            line_list.append(list)
    return line_list

#时间处理9/11/2014 00:10:00
def time_resolve(timestr='2014-04-01 00:00:00'):
    #2014-04-01 00:00:00 --》1396281600
    #9/11/2014 00:00:00 -->1410364800.0 +1h-->1410368400=3600s
    try:

        mktime = time.mktime(time.strptime(timestr, '%Y-%m-%d %H:%M:%S'))
    except Exception,ex:
        try:
            mktime = time.mktime(time.strptime(timestr, '%m/%d/%Y %H:%M:%S'))
        except Exception, ex:
            print ex


    mark=(mktime-1396281600)//3600+1  #取9/11/2014 00:10:00 为1，一次类推
    return mark

#处理得到type 类型
def type_get(x,y,line_list):
    type=None
    for i in line_list:
        if i[0]==x and i[1]==y:
            type=i[2]
            break
    return  type
#读取类别文件，保存
line_list=readcluster(clusterpath)

#读取cvs文件
def readcvs(paths):
    start_list=[]
    stop_list=[]
    for  path in paths:#读取所有的路径
        print u'正在读取文件{0}...'.format(path)
    #这两个同级
        with open(path, 'rb') as f:
            reader = csv.reader(f)
            #['1941', '2014-04-01 08:34:13', '2014-04-01 09:06:34', '519', 'E 42 St & Vanderbilt Ave', '40.752416', '-73.97837', '315', 'South St & Gouverneur Ln', '40.70355377', '-74.00670227', '20536', 'Subscriber', '1971', '1']
            for row in reader:
                try:
                    res=[row[1],row[2],row[5],row[6],row[9],row[10]]
                    start_mk=time_resolve(res[0])
                    stop_mk=time_resolve(res[1])
                    if start_mk>0 and start_mk<=4392:
                        # 1、处理条目保存为两类;开始时间，开始类型，结束时间，结束类型
                        start_list.append([start_mk,type_get(res[2],res[3],line_list),stop_mk,type_get(res[4],res[5],line_list)])
                except Exception,ex:
                    print ex
    return start_list
def main():
    sanweis = []
    siweis = []

    startlist = readcvs(cvspathlist)
    for i in range(1,4393):
        sanweis = []
        # sanweis_outout = []
        list_num=[] #一整天的起点数量
        erweis_output = []
        erweis_iutput = []
        for a in startlist:
            if a[0]==i:#起始时间在对应小时文件内
                list_num.append(a)
        #print list_num
        for i1 in range(0,12):#定义起点序列
            list_excel = []
            for i2 in range(0,12):#定义终点序列
                count_num=0
                count_start=0
                cont_stop=0
                for mm in list_num:#遍历一小时的序列
                    if mm[1]==str(i1) :
                        count_start+=1#某一类坐标作为起点的数量
                        if mm[3]==str(i2):
                            cont_stop+=1#某一类坐标作为终点的数量

                #横向写入
                if count_start==0:
                    list_excel.append(count_start)
                    #pass
                else:
                    list_excel.append(float(cont_stop) )
            erweis_output.append(list_excel)
            erweis_input = np.transpose(erweis_output)
        sanweis.append(erweis_output)
        sanweis.append(erweis_input)
        # print '------------------------------------------erweis_output---------------------------------------------------------'
        # print erweis_output
        # print '++++++++++++++++++++++++++++++++++++++++++sanweis++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        # print sanweis
        siweis.append(sanweis)
        # print '===========================================siweis=================================================================='

    np.array(siweis)
    print siweis
    f = h5py.File('data_5_12.h5', 'w')  # 打开h5文件
    f.create_dataset('data', shape=(4392, 2, 12, 12), dtype='float64', data=siweis)


if __name__ == "__main__":
    main()