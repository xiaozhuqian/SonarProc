'''bs, geocoords, heading/course preprocessing'''
import copy
import numpy as np
from scipy import interpolate
import sys
import logging
sys.path.append('.')
import geocoding.geo_proj_conversion as geo_proj_trans

def down_sample(bs, geo_coords, attitudes, nmeas, range, geo_resolution):
    ori_res = range*2/bs.shape[1]
    factor = int(geo_resolution//ori_res)
    bs_down_sample = bs[:, ::factor]
    if bs_down_sample.shape[1] % 2 != 0:
        bs_down_sample = bs_down_sample[:,:-1]
    geo_coords_down_sample = geo_coords
    attitudes_down_sample = attitudes
    nmeas_down_sample = nmeas
    return bs_down_sample, geo_coords_down_sample, attitudes_down_sample, nmeas_down_sample

def remove_adjacent_duplicates(arr):
    '''
    arr: 1/n-d ndarray with repeated  (x,y) in adjent each row
    return:
    result: 1/n-d ndarray, remove adjent repeated row
    index: the kept rows index in arr
    '''
    # 创建一个新的空数组
    result = []
    index = []
    
    # 遍历原始数组
    for i in range(arr.shape[0]):
        # 检查是否与前一个元素相同
        # if i==97:
        if i == 0 :
            # 添加到结果数组中
            result.append(arr[i])
            index.append(i)
        else:
            if not (arr[i] == arr[i-1]).all():
                result.append(arr[i])
                index.append(i)
    # 返回结果数组
    return np.array(result), index

def interpolate_at_repeat(x, x_index):
    '''
        x: 1-d ndarray with adjencent repeated value; eg, array[1,1,2,3,3]
        index: rows index not duplicates in x; eg, [0,2,3]
        return interpolated x at repeated item: 1-d ndarray
        只要给定包含不重复值的一维数组和不重复值在原数组中的索引，即可插值，即
        x=array[1,1,2,3,3], x_index=[0,2,3]的插值结果和x=array[1,0,2,3,0], x_index=[0,2,3]插值结果一致
    '''
    import copy
    new_x = []
    index = copy.deepcopy(x_index)
    index.append(x.shape[0]) #添加最后一个索引，使得插值后的x与原x长度相等
    for j in range(1,len(index)):
        if j != len(index)-1:
            for i in range(index[j-1],index[j]):  
                if i == index[j-1] :
                    new_x.append(x[index[j-1]])
                else:
                    interpolate_value = x[index[j-1]] + (i-index[j-1])*(x[index[j]]-x[index[j-1]])/(index[j]-index[j-1])
                    new_x.append(interpolate_value)
        else: #最后一个间隔
            for i in range(index[j-1],index[j]):   
                if i == index[j-1] :  
                    new_x.append(x[index[j-1]])
                else: #最后一个间隔增幅用上一个间隔
                    interpolate_value = x[index[j-1]] + (i-index[j-1])*(x[index[j-1]]-x[index[j-2]])/(index[j-1]-index[j-2])
                    new_x.append(interpolate_value)

    new_x = np.array(new_x)
    return new_x

def sequence_clean(sequence):
    '''
    sequence: 1-d ndarray
    if 0 exist, fill with its non zero neighbor
    return:
        cleaned sequence
    '''
    cleaned_sequence = copy.deepcopy(sequence)
    if  np.any(sequence == 0):
        abnoraml_index = np.where(sequence==0)
        #logging.info(f'0 sequence index: {abnoraml_index}')
        for i in abnoraml_index[0]:
            if i == 0:
                j = i
                for j in range(np.max(abnoraml_index[0])):
                    if j not in abnoraml_index[0]:
                        value = cleaned_sequence[j]
                        break
                cleaned_sequence[i] = value
            else:
                cleaned_sequence[i] = cleaned_sequence[i-1]
    
    return cleaned_sequence

def sequence_smooth(data, smooth_factor):
    '''
    spline fit
    data: 1d array
    '''
    data_sample, data_index = remove_adjacent_duplicates(data)
    x = np.arange(len(data_sample))
    y = data_sample
    tck,u = interpolate.splprep([x,y],k=3,s=smooth_factor)
    u_new = np.zeros_like(data)
    u_new[data_index] = u
    u_new = interpolate_at_repeat(u_new, data_index)
    out = interpolate.splev(u_new,tck)
    data_interpolate = np.array(out[1])
    return data_interpolate, tck, u_new

def geo_coords_clean(geo_coords):
    '''
    fill the missing geo_coords which is 0 with last non zero row
    geo_coords: n_ping*2, (lon, lat)
    return:
        cleared_geo_coords: n_ping*2, (lon, lat)
    '''
    cleared_geo_coords = copy.deepcopy(geo_coords)
    if  np.any(geo_coords == 0):
        abnoraml_index_1 = np.where(geo_coords[:,0]==0)
        abnoraml_index_2 = np.where(geo_coords[:,1]==0)
        abnoraml_index = list(set(abnoraml_index_1[0])&set(abnoraml_index_2[0]))
        abnoraml_index.sort()
        #logging.info(f'0 geocoords index: {abnoraml_index}')
        for i in abnoraml_index:
            if i == 0:
                j = i
                for j in range(np.max(abnoraml_index)+2):
                    if j not in abnoraml_index:
                        value = cleared_geo_coords[j]
                        break
                cleared_geo_coords[i] = value
            else:
                cleared_geo_coords[i] = cleared_geo_coords[i-1]

    return cleared_geo_coords

def fish_geocoords2fish_spatialcoords(fish_geocoords,geo_EPSG,proj_EPSG):
    '''
    fish_geocoords: [lon, lat] with degree unit, ping_count*2
    geo_EPSG: geographic coordinates system
    proj_EPSG: projection coordinates system
    fsih_spatialcoords: [x,y], x corresponds to lon, y correspons to lat, ping_count*2
    '''
    fish_lon = fish_geocoords[:,0]
    fish_lat = fish_geocoords[:, 1]

    fish_spatialcoords = []
    for i in range(0, fish_geocoords.shape[0]):
        longitude = fish_lon[i]
        latitude = fish_lat[i]
        fish_x, fish_y = geo_proj_trans.geo2proj(latitude, longitude, geo_EPSG=geo_EPSG, proj_EPSG=proj_EPSG)
        fish_spatialcoord = np.array([fish_x, fish_y, 0])
        fish_spatialcoord = fish_spatialcoord.reshape(1,fish_spatialcoord.shape[0])
        fish_spatialcoords.append(fish_spatialcoord)
    
    return np.array(fish_spatialcoords).squeeze()

def coords_smooth(fishcoord, smooth_type, smooth_factor):
    '''
    linear or cubic spline line interpolate.
    input:
        geocoord.shape: ping_count*2, (x(lon),y(lat))
    return:
        x_interpolate.shape: (ping_count,)
        y_interpolate.shape: (ping_count,)
    '''
    geocoord_x = fishcoord[:, 0]
    geocoord_y = fishcoord[:, 1]

    fish_gps_x_sample, fish_gps_x_index = remove_adjacent_duplicates(fishcoord[:])
    if smooth_type == 'linear': #相当于bspline中的interpolate.splprep([x_sample,y_sample],k=1,s=0)
        x_interpolate = interpolate_at_repeat(geocoord_x, fish_gps_x_index)
        y_interpolate = interpolate_at_repeat(geocoord_y, fish_gps_x_index)
        tck = None
        u_new = None

    if smooth_type == 'bspline':
        x_sample = fish_gps_x_sample[:,0]
        y_sample = fish_gps_x_sample[:,1]

        tck,u = interpolate.splprep([x_sample,y_sample],k=3,s=smooth_factor)

        u_new = np.zeros_like(geocoord_x)
        u_new[fish_gps_x_index] = u
        u_new = interpolate_at_repeat(u_new, fish_gps_x_index)
        
        out = interpolate.splev(u_new,tck)
        x_interpolate = np.array(out[0])
        y_interpolate = np.array(out[1])

    return x_interpolate, y_interpolate, tck, u_new

def sequence_prep(sequence, smooth_factor):
    '''for 1d array'''
    cleaned_sequence = sequence_clean(sequence)
    smoothed_sequence, sequence_tck, sequence_u = sequence_smooth(cleaned_sequence, smooth_factor)
    return smoothed_sequence, sequence_tck, sequence_u

def coords_prep(coords, smooth_type, smooth_factor, geo_EPSG,proj_EPSG):
    '''for tow fish geographic coordinates'''
    cleaned_coords = geo_coords_clean(coords)
    fish_spatialcoords = fish_geocoords2fish_spatialcoords(cleaned_coords,geo_EPSG,proj_EPSG)
    x_smoothed, y_smoothed, coords_tck, coords_u = coords_smooth(fish_spatialcoords, smooth_type, smooth_factor)
    
    return np.array([x_smoothed, y_smoothed]).T, coords_tck, coords_u 


if __name__ == "__main__":
    coords = []
    with open('./data/deep1_20-nmea.txt') as f:
        for line in f:
            port_x = float(line.strip().split(' ')[0])
            coords.append(port_x)
    coords = np.array(coords)
    smoothed,_,_ = sequence_smooth(coords,10)
    np.savetxt('./nmea.txt', coords, fmt='%.1f')
    np.savetxt('./smoothed.txt', smoothed, fmt='%.3f')
    print(coords)
    print(smoothed)





