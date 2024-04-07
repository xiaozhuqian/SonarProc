'输入的单个文件必须量程不变'

import argparse
import os
import scipy.io as io
import numpy as np
import cv2
import logging
import sys
sys.path.append('.')
import utils.logger as logger
from utils.vis import gray_enhancing_tvg, linearStretch
from mat2npy import extract_from_mat
from prep.preprocessing import down_sample
from utils.util import absoption


def parse_args():
    parser = argparse.ArgumentParser(description='gen_image')
    parser.add_argument('--input_dir', default='D:/SidescanData_202308/new1/mat_1/', help='directory of input file')
    parser.add_argument('--out_dir_raw', default='D:/SidescanData_202308/new1/npy/raw/', help='dierctory to save raw npy output')
    parser.add_argument('--out_dir_downsample', default='D:/SidescanData_202308/new1/npy/downsample/', help='dierctory to save downsampled npy and png')
    
    parser.add_argument('--frequency', default=20, type=int, help='high frequency=21, low frequency = 20')
    parser.add_argument('--range', default=35., type=float, help='range of port or starboard')
    
    parser.add_argument('--temperature', default=26., type=float, help='slant_range')
    parser.add_argument('--salinity', default=28., type=float, help='slant_range')
    parser.add_argument('--depth', default=5., type=float, help='slant_range')
    parser.add_argument('--pH', default=8., type=float, help='slant_range')
    parser.add_argument('--lambd', default=20., type=float, help='tvg log factor')
    parser.add_argument('--ratio', default=0.02, type=float, help='gray linear strech clip ratio')
    parser.add_argument('--geo_resolution', default=0.1, type=float, help='downsamplinged img resolution')

    args = parser.parse_args()
    return args

def main(args):

    input_names = []
    for filepath, dirnames, filenames in os.walk(args.input_dir):
        for filename in filenames:
            input_names.append(filename)
    
    #input_names = ['line155_004.mat']
    for idx, input_name in enumerate(input_names):
        input_path = os.path.join(args.input_dir, input_name)
        index = input_name.split('.')[-2]
        mat = io.loadmat(input_path)

        sonardata, geo_coords, attitudes, nmeas, remain_acoustics, error_ping_number \
            =extract_from_mat(mat, args.frequency)
        
        sonardata_downsample, _, _, _ = down_sample(sonardata, geo_coords, attitudes, nmeas, \
                                                    args.range, args.geo_resolution)
        
        # tvg enhancement
        if args.frequency == 20:
            frequency = 600
        if args.frequency == 21:
            frequency = 1600
        a = absoption(frequency*1000., args.temperature, args.salinity, args.depth, args.pH) # sound absoption in seawater, db/m
        sonardata_tvg = gray_enhancing_tvg(sonardata_downsample, args.range, args.lambd, a)
        img = linearStretch(sonardata_tvg, 1, 255, args.ratio)

        logging.info(f'{idx}th image/{len(input_names)}, {input_name}, shape_raw: {sonardata.shape}, shape_downsample: {sonardata_downsample.shape}, remains: {remain_acoustics}, error_ping: {error_ping_number}') 
        
        #np.save(os.path.join(args.out_dir_raw, index+'_'+str(args.frequency)+'-bs'+'.npy'), sonardata)
        np.save(os.path.join(args.out_dir_raw, index+'_'+str(args.frequency)+'-geo_coords'+'.npy'), geo_coords)
        np.save(os.path.join(args.out_dir_raw, index+'_'+str(args.frequency)+'-attitudes'+'.npy'), attitudes)
        np.save(os.path.join(args.out_dir_raw, index+'_'+str(args.frequency)+'-nmeas'+'.npy'), nmeas)

        np.save(os.path.join(args.out_dir_downsample, index+'_'+str(args.frequency)+'-bs'+'.npy'), sonardata_downsample)
        cv2.imwrite(os.path.join(args.out_dir_downsample, index+'_'+str(args.frequency)+'-bs'+'.png'), img)

if __name__ == '__main__':
    import time
    args = parse_args()

    if not os.path.exists(args.out_dir_raw):
        os.makedirs(args.out_dir_raw)
    if not os.path.exists(args.out_dir_downsample):
        os.makedirs(args.out_dir_downsample) 

    now = int(round(time.time()*1000))
    now02 = time.strftime('%Y-%m-%d-%H%M%S',time.localtime(now/1000))
    logger.setlogger(os.path.join(args.out_dir_raw, f'{now02}-npy_extraction.log'))  # set logger
    for k, v in args.__dict__.items():  # save args
        logging.info("{}: {}".format(k, v))
    
    main(args)
