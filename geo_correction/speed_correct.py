import cv2
import numpy as np
import sys
import time
import logging
import glob
sys.path.append('.')
import prep.preprocessing as prepp
import utils.logger as logger
from QC.quantity_results import cal_survey_line_length

def blockwise_speed_correction(bs, geo_coords, resolution, geo_EPSG, proj_EPSG):
    geo_coords = prepp.geo_coords_clean(geo_coords) # fill the missing geo coords
    fish_spatialcoords = prepp.fish_geocoords2fish_spatialcoords(geo_coords,geo_EPSG,proj_EPSG) # fish geo coords projection
    fish_gps_x_sample, fish_gps_x_index = prepp.remove_adjacent_duplicates(fish_spatialcoords[:])
    
    speed_corrected_bs = []
    #bs_reshaped_shape = []
    for i, index in enumerate(fish_gps_x_index):
        if i >=1:
            index_pre = fish_gps_x_index[i-1]
            fish_x_pre = fish_spatialcoords[index_pre,0]
            fish_y_pre = fish_spatialcoords[index_pre,1]
            fish_x = fish_spatialcoords[index,0]
            fish_y = fish_spatialcoords[index,1]
            dis = np.sqrt((fish_x-fish_x_pre)**2+(fish_y-fish_y_pre)**2)
            bs_segment = bs[index_pre:index,:]
            h_o = bs_segment.shape[0]
            h_r = int(dis/resolution)
            inter_ratio = h_r/h_o
            inter_fold = int(np.trunc(inter_ratio))
            if inter_fold >= 1: #interpolate pings
                remain_count = h_r - inter_fold*h_o
                random_inter_index = sorted(np.random.choice(h_o, remain_count, replace=False))
                random_inter_index_reshape = [ele*inter_fold for ele in random_inter_index]
                bs_reshaped = bs_segment.repeat(inter_fold,axis=0)
                bs_reshaped = np.insert(bs_reshaped, random_inter_index_reshape,bs_segment[random_inter_index],axis=0)
            if inter_fold == 0: # delete pings
                random_del_index = sorted(np.random.choice(h_o, h_o-h_r, replace=False))
                bs_reshaped = np.delete(bs_segment,random_del_index,axis=0)
            #bs_reshaped_shape.append(len(speed_corrected_bs))
 
            for ping in bs_reshaped:
                speed_corrected_bs.append(ping)
    # print(fish_gps_x_index)
    # print(bs_reshaped_shape)
    return np.array(speed_corrected_bs)


def overall_speed_correction(bs, geo_coords, resolution, geo_EPSG, proj_EPSG):
    geo_coords = prepp.geo_coords_clean(geo_coords) # fill the missing geo coords
    fish_spatialcoords = prepp.fish_geocoords2fish_spatialcoords(geo_coords,geo_EPSG,proj_EPSG) # fish geo coords projection
    fish_gps_x_sample, fish_gps_x_index = prepp.remove_adjacent_duplicates(fish_spatialcoords[:])
    proj_x = (fish_spatialcoords[fish_gps_x_index])[:,0]
    proj_y = (fish_spatialcoords[fish_gps_x_index])[:,1]
    survey_line_length = cal_survey_line_length(proj_x, proj_y)
    
    width = bs.shape[1]
    height_resize = int(survey_line_length/resolution)

    resized = cv2.resize(bs, (width, height_resize),interpolation=cv2.INTER_AREA)
  
    return resized

if __name__ == "__main__":
    import argparse
    import os
    import time
    parser = argparse.ArgumentParser(description='detect sea bottom with gradient-mean-abnormal_detection-smooth')
    parser.add_argument('--root', default='D:/SidescanData_202312/sanshan_80', help='root')
    parser.add_argument('--save_file', default='speed_correction', help='file name of saved file')
    parser.add_argument('--speed_correction_method', default='blockwise', help='blockwise, overall')
    parser.add_argument('--res', default=0.1, type=float, help='geocoded image resolution')
    parser.add_argument('--geo_EPSG', default=4490, type=int, help='geographic coordinate system')
    parser.add_argument('--proj_EPSG', default=4499, type=int, help='projection coordinate system')

    args = parser.parse_args()

    root = args.root
    save_file = args.save_file
    
    if not os.path.exists(os.path.join(root, 'tile', save_file)):
        os.makedirs(os.path.join(root, 'tile', save_file)) 

    now = int(round(time.time()*1000))
    now02 = time.strftime('%Y-%m-%d-%H%M%S',time.localtime(now/1000))
    logger.setlogger(os.path.join(root, 'tile', save_file, f'{now02}-geocoding_coarse2fine.log'))  # set logger
    for k, v in args.__dict__.items():  # save args
        logging.info("{}: {}".format(k, v))

    slant_bs_names = []
    for slant_bs_path in sorted(glob.glob(os.path.join(root,'npy','downsample', '*-bs.npy'))):
        _, slant_bs_name = os.path.split(slant_bs_path)
        slant_bs_names.append(slant_bs_name)

    #slant_bs_names = ['two1_20-bs.npy']
    for n, slant_name in enumerate(slant_bs_names):
        # read variables
        name_index = slant_name.split('.')[-2] 
        flat_bs = cv2.imread(os.path.join(root, 'tile', 'gray_enhancement_coarse2fine',f'{name_index}-grayenhance.png'),0) #(ping_count, flat_sample_count)
        geo_coords = np.load(os.path.join(root, 'npy', 'raw',slant_name).replace('bs','geo_coords')) #(ping_count, 2), (lon,lat) with unit degree
        geo_coords = geo_coords[:,:2]

        logging.info(f'{n}/{len(slant_bs_names)}, {name_index}')
        start = time.time()
        if args.speed_correction_method == 'blockwise':
            speed_corrected_bs = blockwise_speed_correction(flat_bs, geo_coords, args.res, args.geo_EPSG, args.proj_EPSG)
        if args.speed_correction_method == 'overall':
            speed_corrected_bs = overall_speed_correction(flat_bs, geo_coords, args.res, args.geo_EPSG, args.proj_EPSG)
        cost_time = time.time()-start

        logging.info(f'{n}/{len(slant_bs_names)}, {name_index}, time_cost: {cost_time:.3f} s')

        cv2.imwrite(os.path.join(root, 'tile',save_file,f'{name_index}-speed_correction-entire.png'), speed_corrected_bs)

    


