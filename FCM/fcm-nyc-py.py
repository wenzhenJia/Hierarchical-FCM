# %%
import math
import os
import time
from math import cos, sin

import numpy as np
import pandas as pd
from fcmeans import FCM
from sklearn.preprocessing import MinMaxScaler


# %%
# 时间处理9/11/2014 00:10:00
def time_resolve(timestr='2014-04-01 00:00:00'):
    #2014-04-01 00:00:00 --》1396281600
    #9/11/2014 00:00:00 -->1410364800.0 +1h-->1410368400=3600s
    try:
        mktime = time.mktime(time.strptime(timestr, '%Y-%m-%d %H:%M:%S'))
    except Exception as ex:
        try:
            mktime = time.mktime(time.strptime(timestr, '%m/%d/%Y %H:%M:%S'))
        except Exception as ex:
            print(ex)
    mark=(mktime-1396281600)//3600+1  #取9/11/2014 00:10:00 为1，一次类推
    return int(mark)
 
 
def rad(d):
    return d * math.pi / 180.0
 
def getDistance(lat1, lng1, lat2, lng2):
    # km
    EARTH_REDIUS = 6378.137
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(sin(a/2), 2) + cos(radLat1) * cos(radLat2) * math.pow(sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s

# %%
getDistance(40.71117444,-73.99682619,40.71729,-73.996375)

# %%
n_clusters= 50
time_steps = 4392
stations=pd.read_table('./nyc.txt',sep='\t',header=None).values     #读入txt文件，分隔符为\t
station_num = stations.shape[0]
total_iteration = 11
# total_iteration = 6ss

# %%
def fcm_classify(data, n_clusters, iteration_times=0):
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(data)
    fcm_labels = fcm.predict(data)
    fcm_labels = np.expand_dims(fcm_labels, axis=1)
    station_labels = np.concatenate((stations, fcm_labels), axis=1)
    if not os.path.exists('./cluster-result-{}'.format(n_clusters)):
        os.mkdir('./cluster-result-{}'.format(n_clusters))
    with open("./cluster-result-{}/cluster_{}.txt".format(n_clusters,iteration_times), 'w') as f:
        for row in station_labels:
            f.write('{}\t{}\t{}\n'.format(row[0], row[1], int(row[2])))
    # debug
    # station_labels = pd.read_table('./cluster-new.txt',sep='\t',header=None).values
    # debug
    return station_labels

# %%
NYCdata_dir = list(map(lambda x : os.path.join(os.getcwd(), 'NYCdata', x), os.listdir('./NYCdata')))
for single_csv in NYCdata_dir:
    single_data_frame = pd.read_csv(single_csv)
    if single_csv == NYCdata_dir[0]:
        all_data_frame = single_data_frame
    else:  # concatenate all csv to a single dataframe, ingore index
        all_data_frame = pd.concat([all_data_frame, single_data_frame], ignore_index=True)
all_data_frame['starttime'] = all_data_frame['starttime'].transform(time_resolve)
all_data_frame['stoptime'] = all_data_frame['stoptime'].transform(time_resolve)

# %%
all_data_frame = all_data_frame.values

# %%
station_label_idx = dict()
for idx, station_label in enumerate(stations):
    station_label_idx[str(station_label[0])+str(station_label[1])] = idx

latlng_matrix = np.zeros((station_num, station_num))
time_distance_matrix = np.zeros(((station_num, station_num)))

for i in range(station_num):
    for j in range(station_num):
        latlng_matrix[i][j] = getDistance(stations[i][0], stations[i][1], stations[j][0], stations[j][1])

for row in all_data_frame:
    start_station = str(row[5])+str(row[6])
    end_station = str(row[9]) + str(row[10])
    if start_station not in station_label_idx or end_station not in station_label_idx: continue
    # starttime 
    if row[1] > time_steps or row[2] > time_steps: continue
    i = station_label_idx[start_station]
    j = station_label_idx[end_station]
    if row[0] > time_distance_matrix[i][j] and time_distance_matrix[i][j] != 0:
        pass
    time_distance_matrix[i][j] = row[0]

# %%
latlng_matrix_norm = np.zeros((station_num))
time_distance_matrix_norm = np.zeros((station_num))
for i in range(station_num):
    latlng_matrix_norm[i] = np.linalg.norm(latlng_matrix[i, :])
    time_distance_matrix_norm[i] = np.linalg.norm(time_distance_matrix[i, :])
min_max_scaler_norm_1 = MinMaxScaler()
latlng_matrix_norm = min_max_scaler_norm_1.fit_transform(latlng_matrix_norm.reshape(-1, 1))
min_max_scaler_norm_2 = MinMaxScaler()
time_distance_matrix_norm = min_max_scaler_norm_2.fit_transform(time_distance_matrix_norm.reshape(-1, 1))
road_network_information = np.concatenate((latlng_matrix_norm, time_distance_matrix_norm), axis=1)

# %%
# # 第一次根据欧氏距离和骑行时间计算
# station_labels = fcm_classify(road_network_information, n_clusters)
# station_label_map = dict()
# for station_label in station_labels:
#     station_label_map[str(station_label[0])+str(station_label[1])] = int(station_label[2])

def cal_station_label_map_idx(data, n_clusters, i):
    # 迭代根据六维进行分类
    station_labels = fcm_classify(data, n_clusters, i)
    station_label_map = dict()
    for station_label in station_labels:
        station_label_map[str(station_label[0])+str(station_label[1])] = int(station_label[2])
    return station_label_map

# %%
def cal_extra_information(all_data_frame, station_label_idx, station_label_map):
    road_in = dict()
    road_out = dict()
    trend_in = dict()
    trend_out = dict()
    for row in all_data_frame:
        start_station = str(row[5])+str(row[6])
        end_station = str(row[9]) + str(row[10])
        if start_station not in station_label_idx or end_station not in station_label_idx: continue
        # starttime 
        if row[1] > time_steps or row[2] > time_steps: continue
        # 每个小时该站点到每个站点流入矩阵、流出矩阵
        if start_station in road_out:
            road_out[start_station][row[1]-1, station_label_idx[end_station]] += 1
        else:
            road_out[start_station] = np.zeros((time_steps,station_num))
        if end_station in road_in:
            road_in[end_station][row[2]-1, station_label_idx[start_station]] += 1
        else:
            road_in[end_station] = np.zeros((time_steps,station_num))
        # 每个小时该站点到每个簇流入矩阵、流出矩阵
        if start_station in trend_out:
            trend_out[start_station][row[1]-1, station_label_map[end_station]] += 1
        else:
            trend_out[start_station] = np.zeros((time_steps,station_num))
        if end_station in trend_in:
            trend_in[end_station][row[2]-1, station_label_map[start_station]] += 1
        else:
            trend_in[end_station] = np.zeros((time_steps,station_num))
    station_with_extra_information = np.concatenate((stations, np.zeros((station_num, 4)) ), axis=1)
    for station in station_with_extra_information:
        station_name = str(station[0])+str(station[1])
        # 流入矩阵、流出矩阵 求斐波那契范数
        station[2] = np.linalg.norm(road_in[station_name])
        station[3] = np.linalg.norm(road_out[station_name])
        station[4] = np.linalg.norm(trend_in[station_name])
        station[5] = np.linalg.norm(trend_out[station_name])
    extra_information = station_with_extra_information[:, 2:4]
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler()
    extra_information = min_max_scaler.fit_transform(extra_information)
    # station_with_normal_extra_information = np.concatenate((stations, extra_information, station_with_extra_information[:,4:]), axis=1)
    station_with_normal_extra_information = np.concatenate((road_network_information, station_with_extra_information[:,4:]), axis=1)
    return station_with_normal_extra_information

# %%
stations_data = road_network_information
for i in range(1, total_iteration+1):
    station_label_map = cal_station_label_map_idx(stations_data, n_clusters, i)
    station_with_normal_extra_information = cal_extra_information(all_data_frame, station_label_idx, station_label_map)
    stations_data = station_with_normal_extra_information
    # with open("./cluster-result/cluster_info.txt", 'w') as f:
    #     for row in stations_data:
    #         f.write('{}\t{}\t{}\t{}\n'.format(row[0], row[1], row[2], row[3]))

# %%
# stations_data

# %%



