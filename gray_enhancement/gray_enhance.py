import numpy as np
from scipy.signal import savgol_filter
import cv2
import matplotlib.pyplot as plt
import scipy.signal
import sys
sys.path.append('.')
from utils.vis import linearStretch

def gray_enhancing_statistic(bs_geocorrection, min_port_width, min_starboard_width, ratio=0.02):
    n_sample = bs_geocorrection.shape[1]
    n_ping = bs_geocorrection.shape[0]
    port = np.fliplr(bs_geocorrection[:,0:n_sample//2])
    starboard = bs_geocorrection[:,(n_sample//2):]
    min_width = min(min_port_width, min_starboard_width)
    
    bs_dic = {'port':port, 'starboard': starboard}
    factor_dic = {'port': port[0], 'starboard':starboard[0]}
    for (key, value) in bs_dic.items():
        bs_value = value
        bs_pingsec_mean = np.mean(bs_value, axis=0)
        bs_pingsec_mean[min_width:] = bs_pingsec_mean[min_width-1]
        bs_mean = np.mean(bs_value)
        factor = bs_mean/bs_pingsec_mean
        factor_dic[key] = factor

    factor_dual = np.concatenate((factor_dic['port'][::-1], factor_dic['starboard']),axis=0)
    factor_dual_smooth = savgol_filter(factor_dual, 51, 2, mode= 'nearest')
    bs_corrected = bs_geocorrection*factor_dual_smooth
    #bs_corrected_clip = np.uint8(np.minimum(np.maximum(bs_corrected, 0), 255))
    bs_corrected_clip = linearStretch(bs_corrected, 1, 255, ratio)

    return bs_corrected, bs_corrected_clip

def gray_enhancing_retinex(bs_geocorrection, gaussian_kernel=300, A=100., alpha=2.):
    bs_gaussian = cv2.GaussianBlur(bs_geocorrection, (0,0), gaussian_kernel)
    gaussian_corrected = A*bs_geocorrection/(bs_gaussian+alpha)
    gaussian_corrected_clip = np.uint8(np.minimum(np.maximum(gaussian_corrected, 0), 255))
    return gaussian_corrected, gaussian_corrected_clip

def no_object_ping_detect(enhanced_bs, prob_thred=0.3):
    no_object_pings = []
    ping_indexs = []
    for i, ping in enumerate(enhanced_bs):
        ping = savgol_filter(ping, 11, 3, mode= 'nearest')
        px, _ = scipy.signal.find_peaks(ping, prominence=50)
        peak = ping[px]
        var = np.var(ping)
        mean = np.mean(ping)
        if peak.shape[0]==0:
            no_object_pings.append(ping)
            ping_indexs.append(i)
            continue
        prob = mean/max(peak) # Markov's inequality for abnormal detection
        # prob_1 = var/((max(peak)-mean) ** 2)
        # dist = max(peak)-mean
        if prob > prob_thred: #prob=0.3, prob_1 = 0.05
            no_object_pings.append(ping)
            ping_indexs.append(i)

    return no_object_pings, ping_indexs

def cal_factor(bs,min_port_width, min_starboard_width):
    n_sample = bs.shape[1]
    n_ping = bs.shape[0]
    port = np.fliplr(bs[:,0:n_sample//2])
    starboard = bs[:,(n_sample//2):]
    min_width = min(min_port_width, min_starboard_width)
    
    bs_dic = {'port':port, 'starboard': starboard}
    factor_dic = {'port': port[0], 'starboard':starboard[0]}
    for (key, value) in bs_dic.items():
        bs_value = value
        bs_pingsec_mean = np.mean(bs_value, axis=0)
        bs_pingsec_mean[min_width:] = bs_pingsec_mean[min_width-1] # necessary if abnormal values at edge exit, for sanshan data
        bs_mean = np.mean(bs_value)
        factor = bs_mean/bs_pingsec_mean
        factor_dic[key] = factor

    factor_dual = np.concatenate((factor_dic['port'][::-1], factor_dic['starboard']),axis=0)
    factor_dual_smooth = savgol_filter(factor_dual, 51, 2, mode= 'nearest')

    return factor_dual_smooth

def coarse2fine(bs, min_port_width, min_starboard_width, prob_thred, ratio):
    coarse_factor = cal_factor(bs, min_port_width, min_starboard_width)
    coarse_bs = bs*coarse_factor*10.
    _, ping_index = no_object_ping_detect(coarse_bs, prob_thred)
    fine_pings = bs[ping_index,:]
    fine_factor = cal_factor(fine_pings,min_port_width, min_starboard_width)
    fine_bs = bs*fine_factor
    #fine_bs = logTransform(fine_bs, 200, 256)
    #fine_bs = gamaTransform(fine_bs, c=256, r=0.5)

    # plt.figure()
    # x = np.arange(0,len(coarse_factor))
    # plt.plot(x,coarse_factor,label='coarse')
    # plt.plot(x,fine_factor,label='fine')
    # plt.legend()
    # plt.savefig('./data/outputs6/line68_21.png')
    # plt.show()

    fine_bs_strech = linearStretch(fine_bs, 1, 255, ratio)
    no_object_pings = fine_bs_strech[ping_index, :]

    return fine_bs, fine_bs_strech, no_object_pings


if __name__ == '__main__':
    import argparse
    import os
    import glob
    import sys
    sys.path.append('.')
    import utils.logger as logger
    import cv2
    import logging
    import time

    parser = argparse.ArgumentParser(description='detect sea bottom with gradient-mean-abnormal_detection-smooth')
    parser.add_argument('--root', default='D:/SidescanData_202308/southsea_road/', help='directory of input file')
    parser.add_argument('--out_dir', default='D:/SidescanData_202308/southsea_road/tile/gray_enhancement_statistic/', help='dierctory to save image output')
    parser.add_argument('--gray_enhance_method', default='statistic', help='retinex, statistic, coarse2fine')
    parser.add_argument('--prob_thred', default=0.3, type=float, help='no object pings detect probability threshold')
    parser.add_argument('--ratio', default=0.01, type=float, help='gray linear strech clip ratio')
    parser.add_argument('--gaussian_kernel', default=50, type=int, help='gaussian filter kernel') # for retinex gray enhancement
    parser.add_argument('--gain', default=100., type=float, help='multiplication factor') # for retinex gray enhancement
    parser.add_argument('--alpha', default=0.05, type=float, help='intensity equalization factor') # for retinex gray enhancement
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)  

    now = int(round(time.time()*1000))
    now02 = time.strftime('%Y-%m-%d-%H%M%S',time.localtime(now/1000))
    logger.setlogger(os.path.join(args.out_dir, f'{now02}-gray_enhancement.log'))  # set logger
    for k, v in args.__dict__.items():  # save args
        logging.info("{}: {}".format(k, v))

    
    geocorrected_bs_names = []
    for bs_path in sorted(glob.glob(os.path.join(args.root, 'tile', 'geo_correction', '*.png'))):
        _, bs_name = os.path.split(bs_path)
        geocorrected_bs_names.append(bs_name)

    #geocorrected_bs_names = ['line68_20-bs-geocorrection.png']

    for n, bs_name in enumerate(geocorrected_bs_names):
        index = bs_name.split('-')[0]
       
        bs = cv2.imread(os.path.join(args.root, 'tile', 'geo_correction', bs_name),0)

        n_sample = bs.shape[1]
        bs_slant = np.load(os.path.join(args.root, 'npy', 'downsample', f'{index}-bs.npy'))
        w_slant = bs_slant.shape[1]

        if os.path.exists(os.path.join(args.root, 'tile','bottom_line',  f'{index}-bs-bottom_4inter.txt')): #check if inter coords exist
            coord_name = f'{index}-bs-bottom_4inter.txt'
        else:
            coord_name = f'{index}-bs-bottom_3smooth.txt'

        coords = []
        with open(os.path.join(args.root,'tile',  'bottom_line', coord_name)) as f:
            for line in f:
                port_x = int(line.strip().split(' ')[0])
                starboard_x = int(line.strip().split(' ')[1])
                port_y = int(line.strip().split(' ')[2])
                coords.append([port_x,starboard_x, port_y])
        coords = np.array(coords)

        min_port_width = min(coords[:,0])
        min_starboard_width = w_slant - max(coords[:,1])

        logging.info(f'{n}/{len(geocorrected_bs_names)}, {bs_name}')

        if args.gray_enhance_method == 'coarse2fine':
            bs_gray_enhancement, bs_gray_enhancement_vis, no_object_pings = coarse2fine(bs, min_port_width, min_starboard_width, args.prob_thred,args.ratio)
        if args.gray_enhance_method == 'retinex':
            bs_gray_enhancement, bs_gray_enhancement_vis = gray_enhancing_retinex(bs, args.gaussian_kernel, args.gain, args.alpha)
        if args.gray_enhance_method == 'statistic':
            bs_gray_enhancement, bs_gray_enhancement_vis = gray_enhancing_statistic(bs, min_port_width, min_starboard_width, ratio=args.ratio)
       
       
        #save
        cv2.imwrite(os.path.join(args.out_dir, f'{index}-bs-grayenhance.png'), bs_gray_enhancement_vis)
        #cv2.imwrite(os.path.join(args.out_dir, f'{index}-no_object_pings.png'), no_object_pings)
        #np.save(os.path.join(args.out_dir, f'{index}-bs-grayenhance.npy'), bs_gray_enhancement)
    
    