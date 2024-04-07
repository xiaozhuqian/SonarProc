import numpy as np
import cv2

def cal_depth_flat_range(slant_img_width, bottom_coords, slant_range):
    sample_count = slant_img_width
    resolution = slant_range/(sample_count/2)
    port_bottom_coord = sample_count/2 - bottom_coords[:,0]
    starboard_bottom_coord = bottom_coords[:,1] - sample_count/2
    port_bottom_depth = port_bottom_coord *  resolution*-1
    starboard_bottom_depth = starboard_bottom_coord * resolution*-1
    flat_range = np.sqrt(slant_range**2-port_bottom_depth**2)+\
    np.sqrt(slant_range**2-starboard_bottom_depth**2)
    return port_bottom_depth, starboard_bottom_depth, flat_range

def hist_diff(hist_denstity, normal_density):#absolute difference between histgram and normal distribution
    difference = np.sum(np.abs(hist_denstity-normal_density))
    return difference

def cal_survey_line_length(x,y):
    '''
    x: projected x, corresponding to lontitude
    y: projected y, corresponding to latitude
    return:
        length (meter)
    '''
    length_list = []
    length_list = [np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) for i in range(1,len(x))]

    length = sum(length_list)

    return length

def extract_fish_geocoords_range(fish_lon,fish_lat):
    min_lon = np.min(fish_lon)
    max_lon = np.max(fish_lon)
    min_lat = np.min(fish_lat)
    max_lat = np.max(fish_lat)
    return min_lon, max_lon, min_lat, max_lat

def cal_area(bs_geocoding, resolution):
    '''
    bs_geocoding: 1-channel geocoded image with grayscale in 0-255
    resolution: meter/pixel (resolution of x and y are the same)
    '''
    _,thresh = cv2.threshold(bs_geocoding,1,255,0) #larger than 1, set to 255
    area = cv2.countNonZero(thresh) * resolution* resolution

    return area

def cal_statistic(x):
    if x is not None:
        u = round(np.mean(x),1)
        std = round(np.std(x),1)
        max_value = round(x.max(),1)
        min_value = round(x.min(),1)
    else:
        u=None
        std = None
        max_value = None
        min_value = None
    return u, std, max_value, min_value


