import numpy as np

def geo_correction(bs, bottom_coords, offset=[0,2]):
    '''
    corrected image resolution is equal to the slant image.
    input:
        bs: n_ping*n_sample
        bottom_coords: n_ping*3, (port_x, starboard_x, y)
        offset: [port_offset, starboard_offset]
    return:
        corrected image
    '''
    
    n_sample = bs.shape[1]
    n_ping = bs.shape[0]

    port = np.fliplr(bs[:,0:bs.shape[1]//2])
    starboard = bs[:,(bs.shape[1]//2):]

    coords_port = bottom_coords[:,[0,2]]
    coords_starboard = bottom_coords[:, [1,2]]
    coords_port[:, 0] = n_sample//2 - coords_port[:, 0] + offset[0]
    coords_starboard[:, 0] = coords_starboard[:, 0] - n_sample//2 + offset[1]

    bs_dic = {'port':port, 'starboard': starboard}
    coords_dic = {'port': coords_port, 'starboard':coords_starboard}
    max_corrected_port_width = np.uint(np.floor(np.sqrt((n_sample//2)**2-(np.min(coords_port[:,0]))**2)))
    max_corrected_starboard_width = np.uint(np.floor(np.sqrt((n_sample//2)**2-(np.min(coords_starboard[:,0]))**2)))
    max_width = max(max_corrected_port_width, max_corrected_starboard_width)
    min_corrected_port_width = np.uint(np.floor(np.sqrt((n_sample//2)**2-(np.max(coords_port[:,0]))**2)))
    min_corrected_starboard_width = np.uint(np.floor(np.sqrt((n_sample//2)**2-(np.max(coords_starboard[:,0]))**2)))
    corrected_dic = {'port': np.zeros((n_ping, max_width)), 
                     'starboard':np.zeros((n_ping, max_width))}
    for (channel, value) in bs_dic.items():
        bs_value = bs_dic[channel]
        coord_value = coords_dic[channel]
        corrected = corrected_dic[channel]
        corrected_width = corrected.shape[1]
        for i in range(0, n_ping):
            x_b = coord_value[i,0]
            x_corrected = np.arange(0, corrected_width)
            x_ori = np.uint(np.floor(np.sqrt(x_b**2+x_corrected**2)))
            bs_ping = bs_value[i]
            corrected[i][:(x_ori[x_ori<n_sample//2]).shape[0]] = bs_ping[x_ori[x_ori<n_sample//2]] #give bs to corrected, give 0 if bs not enough


    bs_corrected = np.concatenate((np.fliplr(corrected_dic['port']), corrected_dic['starboard']),axis=1)

    return  bs_corrected, (min_corrected_port_width, min_corrected_starboard_width)

if __name__ == '__main__':
    import argparse
    import os
    import glob
    import sys
    sys.path.append('.')
    from utils.vis import gray_enhancing_tvg, npy2img
    import utils.logger as logger
    import cv2
    import logging
    import time

    parser = argparse.ArgumentParser(description='detect sea bottom with gradient-mean-abnormal_detection-smooth')
    parser.add_argument('--bs_dir', default='D:/SidescanData_202312/new1_70/npy/downsample', help='directory of input file')
    parser.add_argument('--coord_dir', default='D:/SidescanData_202312/new1_70/bottom_line', help='dierctory to save npy output')
    parser.add_argument('--out_dir', default='D:/SidescanData_202312/new1_70/geo_correction', help='dierctory to save image output')
    parser.add_argument('--bottom_offset', default=[2,5], type=int, help='bottom coord offset of starboard, positive is plus')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)  

    now = int(round(time.time()*1000))
    now02 = time.strftime('%Y-%m-%d-%H%M%S',time.localtime(now/1000))
    logger.setlogger(os.path.join(args.out_dir, f'{now02}-geocorrection.log'))  # set logger
    for k, v in args.__dict__.items():  # save args
        logging.info("{}: {}".format(k, v))

    
    bs_names = []
    for bs_path in sorted(glob.glob(os.path.join(args.bs_dir,'*.npy'))):
        _, bs_name = os.path.split(bs_path)
        bs_names.append(bs_name)

    bs_names = ['new19_20-bs.npy']

    for n, bs_name in enumerate(bs_names):
        index = bs_name.split('.')[-2]

        if os.path.exists(os.path.join(args.coord_dir, f'{index}-bottom_4inter.txt')): #check if inter coords exist
            coord_name = f'{index}-bottom_4inter.txt'
        else:
            coord_name = f'{index}-bottom_3smooth.txt'
       
        bs = np.load(os.path.join(args.bs_dir, bs_name))

        coords = []
        with open(os.path.join(args.coord_dir, coord_name)) as f:
            for line in f:
                port_x = int(line.strip().split(' ')[0])
                starboard_x = int(line.strip().split(' ')[1])
                port_y = int(line.strip().split(' ')[2])
                coords.append([port_x,starboard_x, port_y])
        coords = np.array(coords)

        logging.info(f'{n}/{len(bs_names)}, {coord_name}')

        bs_corrected = geo_correction(bs, coords, args.bottom_offset)

        #save
        cv2.imwrite(os.path.join(args.out_dir, f'{index}-geocorrection.png'), bs_corrected*5)
        np.save(os.path.join(args.out_dir, f'{index}-geocorrection.npy'), bs_corrected)


