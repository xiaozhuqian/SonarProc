import numpy as np
import math
import copy
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

def coarse_bottom_detection(bs, gradient_thred_factor):
    '''
    bs: numpy array, sonardata, ping*sample
    gradient_thred_factor: float
    return:
        coords: numpy array, ping*3, [[port_x, starboard_x, y], ...]
    '''
    bs_bottom = copy.deepcopy(bs)
    port = np.fliplr(bs_bottom[:,0:bs_bottom.shape[1]//2])
    starboard = bs_bottom[:,(bs_bottom.shape[1]//2):]
    n_ping = port.shape[0]
    n_sample = bs_bottom.shape[1] // 2
    coord_ports = [] # port all pings bottom coords [(x,y),...]
    coord_starboards = []
    info_port = []      # port all pings and samples [[[gradient, (x,y)],...], ...]
    info_starboard = []
    gradient_threds = [] # port and starboard all pings gradient thresholds, [[port_thred, starboard_thred], ...]
    for i in range(0, n_ping):
        # smooth current ping value to filter noise
        port[i] = savgol_filter(port[i], 11, 3, mode= 'nearest')
        starboard[i] = savgol_filter(starboard[i], 11, 3, mode= 'nearest')

        # calculate gradient of current ping at each x
        info_ping_port = [] #当前ping梯度和对应坐标
        info_ping_starboard = [] #当前ping梯度和对应坐标
        for j in range(1, n_sample-1):
            if i==0:
                gradient_port = (2*port[i, j] +port[i+1, j])-\
                                (2*port[i, j-1] +port[i+1, j-1])
                gradient_starboard = (2*starboard[i, j] +starboard[i+1, j])-\
                                (2*starboard[i, j-1] +starboard[i+1, j-1])
            elif i == n_ping-1:
                gradient_port = (port[i-1, j] + 2*port[i, j])-\
                                (port[i-1, j-1] + 2*port[i, j-1])
                gradient_starboard = (starboard[i-1, j] + 2*starboard[i, j])-\
                                (starboard[i-1, j-1] + 2*starboard[i, j-1])
            else:
                gradient_port = (port[i-1, j] + 2*port[i, j] +port[i+1, j])-\
                                (port[i-1, j-1] + 2*port[i, j-1] +port[i+1, j-1])
                gradient_starboard = (starboard[i-1, j] + 2*starboard[i, j] +starboard[i+1, j])-\
                                (starboard[i-1, j-1] + 2*starboard[i, j-1] +starboard[i+1, j-1])
            info_ping_port.append([gradient_port, (j,i)])
            info_ping_starboard.append([gradient_starboard, (j, i)])
        info_ping_port = np.array(info_ping_port, dtype=object) 
        info_ping_starboard = np.array(info_ping_starboard, dtype=object) 
        
        #calculate threshold, 前5ping (max(gradient)*factor) 均值作为阈值
        if i<4:
            gradient_thred_port = np.max(info_ping_port[:,0])*gradient_thred_factor
            gradient_thred_starboard = np.max(info_ping_starboard[:,0])*gradient_thred_factor
        else: 
            gradient_thred_port = (np.max(info_ping_port[:,0])*gradient_thred_factor+gradient_threds[i-1][0]+gradient_threds[i-2][0]+gradient_threds[i-3][0]+gradient_threds[i-4][0])/5
            gradient_thred_port = (np.max(info_ping_starboard[:,0])*gradient_thred_factor+gradient_threds[i-1][1]+gradient_threds[i-2][1]+gradient_threds[i-3][1]+gradient_threds[i-4][1])/5

        #pick interest coords x larger than threshold
        coord_ping_interest_port = info_ping_port[info_ping_port[:,0]>=gradient_thred_port][:, 1].tolist()
        coord_ping_interest_starboard = info_ping_starboard[info_ping_starboard[:,0]>=gradient_thred_starboard][:,1].tolist()  
        if len(coord_ping_interest_port)==0: #if interest coords is none, use the max gradient coord
            coord_ping_interest_port= [info_ping_port[np.where(info_ping_port == np.max(info_ping_port[:,0]))[0][0]][1]]
        if len(coord_ping_interest_starboard)==0:
            coord_ping_interest_starboard= [info_ping_starboard[np.where(info_ping_starboard == np.max(info_ping_starboard[:,0]))[0][0]][1]]
        
        #pick coords x have minimum distance between starboard and port
        dist = cdist(coord_ping_interest_port,coord_ping_interest_starboard,metric='euclidean')
        coord_ping_port = coord_ping_interest_port[np.where(dist == np.min(dist))[0][0]]
        coord_ping_starboard = coord_ping_interest_starboard[np.where(dist == np.min(dist))[1][0]] 

        # save variables
        coord_ports.append(coord_ping_port)
        coord_starboards.append(coord_ping_starboard)
        info_port.append(info_ping_port)
        info_starboard.append(info_ping_port)
        gradient_threds.append([gradient_thred_port, gradient_thred_starboard])

    # transform coords to full image
    coord_ports = np.array(coord_ports)
    coord_ports[:,0] = n_sample-1-coord_ports[:,0]
    coord_starboards = np.array(coord_starboards)
    coord_starboards[:,0] = n_sample-1+coord_starboards[:,0]
    coords = [[x[0],y[0],y[1]] for x, y in zip(coord_ports,coord_starboards)]
    coords = np.array(coords, dtype=int)
    
    return coords

def coords_abnormal_correct(coords, win_h=10):
    '''
    useage:
        在邻域窗口内检测std,若当前坐标与窗口坐标均值之差大于std,则用窗口坐标均值替代；
        窗口坐标均值=sum(xi*wi)/sum(wi)
    input:
        coords: numpy array, ping*3, [[port_x, starboard_x, y], ...]
        win_h: window height/2
    return:
        coord_ab_correct: numpy array, int, ping*3, [[port_x, starboard_x, y], ...]
    '''
    n_ping = coords.shape[0]
    coord_ab_correct = copy.deepcopy(coords)
    for k in [0,1]: # loop port and starboard
        for i in range(0, n_ping):
            x0 = 0 # mean of the window
            w_sum = 0 # weight sum
            x_neighbors = [] #coords x in the window
            for j in range(1, win_h+1): #window height=2*win_h, current ping in the middle of the window
                w = 1 - j/win_h
                if i-j < 0:
                    x0 = x0 + coords[i+j, k] * w + 0
                    x_neighbors.append(coords[i+j, k])
                    w_sum = w_sum + w
                elif i+j >n_ping-1:
                    x0 = x0 + 0 + coords[i-j,k] * w
                    x_neighbors.append(coords[i-j,k])
                    w_sum = w_sum + w
                else:
                    x0 = x0 + coords[i+j, k] * w + coords[i-j,k] * w
                    x_neighbors.append(coords[i+j, k])
                    x_neighbors.append(coords[i-j,k])
                    w_sum = w_sum + 2*w
            x0 = x0/w_sum
            std = np.std(x_neighbors)
            x = coords[i,k]
            if np.abs(x-x0)<=1*std:
                coord_ab_correct[i,k] = x
            else:
                coord_ab_correct[i,k] = int(math.floor(x0))
    
    return coord_ab_correct

def coords_smoothing(coords, window_length=11, polyorder=3):
    '''
    usage:
        use Savitzky-Golay filter smooth coords
    input:
        coords: numpy array, ping*3, [[port_x, starboard_x, y], ...]
        window_length: odd, larger and smoother
        polyorder:  polynominal factor, smaller and smoother
    output:
        coords_smooth: numpy array, int32, ping*3, [[port_x, starboard_x, y], ...]
    '''
    coords_smooth = copy.deepcopy(coords)
    for k in [0,1]: # loop port and starboard
        x_smooth = savgol_filter(coords[:,k].tolist(), window_length, polyorder, mode= 'nearest') # 使用Savitzky-Golay 滤波器后得到平滑图线
        x_smooth = x_smooth.astype(np.int32)
        coords_smooth[:, k] = x_smooth 

    return coords_smooth

def bottom_detection(bs, gradient_thred_factor, win_h_ab):
    coarse_coords = coarse_bottom_detection(bs, gradient_thred_factor)
    ab_corrected_coords = coords_abnormal_correct(coarse_coords, win_h_ab)
    ab_corrected_coords = coords_abnormal_correct(ab_corrected_coords, win_h_ab)
    smoothed_coords = coords_smoothing(ab_corrected_coords)
    return coarse_coords, ab_corrected_coords, smoothed_coords

if __name__ == '__main__':
    import argparse
    import os
    import glob
    import sys
    sys.path.append('.')
    from utils.vis import gray_enhancing_tvg, npy2img, draw_points_on_image
    from ab_line_detection import ab_line_correction, peak_detection
    import utils.logger as logger
    import cv2
    import logging
    import time

    parser = argparse.ArgumentParser(description='detect sea bottom with gradient-mean-abnormal_detection-smooth')
    parser.add_argument('--input_dir', default='D:/SidescanData_202312/sanshan_80/npy/downsample', help='directory of input file')
    parser.add_argument('--out_dir_bottom_image', default='D:/SidescanData_202312/sanshan_80/bottom_line', help='dierctory to save npy output')
    parser.add_argument('--out_dir_bottom_coords', default='D:/SidescanData_202312/sanshan_80/bottom_line', help='dierctory to save image output')
    
    parser.add_argument('--gradient_thred_factor', default=0.1, type=float, help='coefficient of maximum gradient')
    parser.add_argument('--win_h', default=10, type=int, help='abnormal detection window height/2')
    
    parser.add_argument('--use_ab_line_correction', default=1, type=int, help='abnormal line correction flag, if use, 1, esle 0')
    parser.add_argument('--peak_factor', default=0.7, type=float, help='coefficient of peak detection') # work when use_ab_line_correction=1
    parser.add_argument('--valley_std', default=5, type=int, help='valley detection standard deviation')# work when use_ab_line_correction=1

    args = parser.parse_args()
    # print(args.input_dir, args.out_dir_bottom_image, args.out_dir_bottom_coords)
    # print(os.path.join(args.out_dir_bottom_image, 'bottom_detection.log'))
    # path = ('D:/SidescanData_202312/new1_70/npy/downsample')
    # print(path)

    now = int(round(time.time()*1000))
    now02 = time.strftime('%Y-%m-%d-%H%M%S',time.localtime(now/1000))
    logger.setlogger(os.path.join(args.out_dir_bottom_image, f'{now02}-bottom_detection5.log'))  # set logger
    for k, v in args.__dict__.items():  # save args
        logging.info("{}: {}".format(k, v))
    

    bs_names = []
    for bs_path in sorted(glob.glob(os.path.join(args.input_dir, '*.npy'))):
        _, bs_name = os.path.split(bs_path)
        bs_names.append(bs_name)

    bs_names = ['sanshan5_20-bs.npy']

    for n, bs_name in enumerate(bs_names):
        bs=np.load(os.path.join(args.input_dir, bs_name))
        bs_index = bs_name.split('.')[-2]
        n_sample = bs.shape[1]

        logging.info(f'{n}th image/{len(bs_names)}, {bs_name}, shape: {bs.shape}')
        
        coarse_coords, ab_corrected_coords, smoothed_coords = \
            bottom_detection(bs, args.gradient_thred_factor, args.win_h)
        
        ab_line_inter = False
        if args.use_ab_line_correction:
            #abnormal line detect
            coords_port = smoothed_coords[:, [0,2]]
            peaks = peak_detection(coords_port, n_sample, args.peak_factor)
            
            #abnormal line correction
            if peaks.shape[0]:
                coords_inter = ab_line_correction(smoothed_coords, n_sample, args.peak_factor, args.valley_std)
                ab_line_inter = True
        
        #visualization and save
        bs_tvg = gray_enhancing_tvg(bs, 70., 5.0, 0.05)
        img = npy2img(bs_tvg)
        
        if args.use_ab_line_correction and ab_line_inter:
            coords = [coarse_coords, ab_corrected_coords, smoothed_coords, coords_inter]
            name = ['1coarse', '2ab', '3smooth', '4inter']
        else:
            coords = [coarse_coords, ab_corrected_coords, smoothed_coords]
            name = ['1coarse', '2ab', '3smooth',]

        for i, coord in enumerate(coords):
            port_coord = coord[:, [0,2]].tolist()
            starboard_coord = coord[:, [1,2]].tolist()
            points = port_coord + starboard_coord
            color_img = draw_points_on_image(points, img)
            cv2.imwrite(os.path.join(args.out_dir_bottom_image,f'{bs_index}-bottom_{name[i]}.png'), color_img)
            np.savetxt(os.path.join(args.out_dir_bottom_coords,f'{bs_index}-bottom_{name[i]}.txt'),coord, delimiter=' ',fmt='%d')


   