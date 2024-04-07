import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import time
from scipy import interpolate
import logging
import copy
import sys
sys.path.append('.')
import utils.logger as logger
import prep.preprocessing as prepp
from geocoding.save_raster import save_geotiff #run this file should delete geocoding.
import geocoding.geo_proj_conversion as geo_proj_trans


def pingcoord2towcoord(raw_ping, max_flat_range):
    '''
    pingcoord (in pixel):
        origin: port left
        x-axis: parallel to ping
        y-axis: pointing to previous ping
    towcoord (in meter, same with towcoord2spatialcoord tow coord when roll=pitch=0):
        origin: center of ping
        x-axis: parallel to ping
        y-axis: pointing to previous ping
    input:
        raw_ping.shape: ping_count, flat_sample_count
        max_flat_range: half width of ping in meter
    return:
        towcoord_xt in meter: shape: (ping_count, )
    '''
    res = max_flat_range/(raw_ping.shape[0]//2)
    raw_ping_coord = np.arange(0, raw_ping.shape[0])
    pan_coord = raw_ping_coord - raw_ping.shape[0]//2
    scale_coord = pan_coord*res

    return scale_coord

def towcoord2spatialcoord(ping_towcoord, fish_attitude, fish_spatialcoord):
    '''
    useage:
        rotate of towcoord in m with z axis, ignore roll and pitch, ie,roll=pitch=0
        Note: (王冬青,2021)中考虑roll和pitch，但在求towcoord时未考虑这两者，导致towcoord不正确，在坐标转换时也使得求得的(xs,ys,zs)均不正确
        towcoord (in meter):
            origin: tow fish
            x-axis: point to starboard
            y_axis: parallel to tofish, pointing to forward
            z_axis: right hand
        spatialcoord (in meter):
            origin: 中央子午线
            x-axis: pointing to east
            y-axis: pointing to north
            z-axis: right hand
    input:
        ping_towcoord: corrected_sample_count*3,(xt,yt,zt), yt=0, zt=height
        fish_attitude: (roll, pitch, heading) in degree, +- is the same with the exported attitude from discovery; roll=0,pitch=0
        fish_spatialcoord: 1*3, (xf,yf,zf),zf=0
    return:
        ping_spatialcoord: corrected_sample_count*3, (xs,ys,zs), zs=height
    '''
    cos = np.cos
    sin = np.sin

    roll, pitch, heading = fish_attitude
    roll_rad = roll * (np.pi/180.)
    pitch_rad = pitch* (np.pi/180.)
    heading_rad = heading* (np.pi/180.)
    rotate_x = np.array([[1,0,0],[0,cos(pitch_rad),-sin(pitch_rad)],[0,sin(pitch_rad), cos(pitch_rad)]]) #clockwize is positive for rotation angle
    rotate_y = np.array([[cos(roll_rad),0,-sin(roll_rad)],[0,1,0],[-sin(roll_rad),0,cos(roll_rad)]])
    rotate_z = np.array([[cos(2*np.pi-heading_rad),-sin(2*np.pi-heading_rad),0],[sin(2*np.pi-heading_rad),cos(2*np.pi-heading_rad),0],[0,0,1]])
    ping_spatialcoord = rotate_x @ rotate_y @ rotate_z @ ping_towcoord.T + fish_spatialcoord.T

    return ping_spatialcoord.T

def cal_geocodingimg_size(fish_x, fish_y, max_flat_range, res):
    '''
    input:
        strip_spatialcoord: ping_count*corrected_sample_count*2, 1st channel for xs, 2nd channel for ys
        res: resolution in m/pixel
    return: 
        strip_imgcoord: _*_*2, 1st channel for u, 2nd channel for v, dtype=float
        (H,W): strip_image height and width
    '''

    upperleft_spatial_x = np.min(fish_x) - max_flat_range
    upperleft_spatial_y = np.max(fish_y) + max_flat_range
    lowerright_spatial_x = np.max(fish_x) + max_flat_range
    lowerright_spatial_y = np.min(fish_y) - max_flat_range
    
    H = np.int64(np.ceil((upperleft_spatial_y-lowerright_spatial_y)/res)+10.)
    W = np.int64(np.ceil((lowerright_spatial_x-upperleft_spatial_x)/res)+10.)
    
    return (upperleft_spatial_x, upperleft_spatial_y), (H,W)

def pingspatialcoord2imgcoord(spatial_x,spatial_y,upperleft_spatial_x, upperleft_spatial_y, res):
    '''
    input:
        strip_spatialcoord: ping_count*corrected_sample_count*2, 1st channel for xs, 2nd channel for ys
        res: resolution in m/pixel
    return: 
        strip_imgcoord: _*_*2, 1st channel for u, 2nd channel for v, dtype=float
        (H,W): strip_image height and width
    '''
    
    u = (spatial_x - upperleft_spatial_x)/res
    v = (upperleft_spatial_y - spatial_y)/res
    
    return u, v

def cal_max_flat_range(bottom_coord_port, slant_range, slant_img_width):
    '''
    calculate 1/2 flat img width in m
    slant_img_width in pixel
    '''
    min_altitude = slant_range - np.max(bottom_coord_port[:,0]) * (slant_range/slant_img_width)
    max_flat_range = np.sqrt(slant_range**2-min_altitude**2)
    return max_flat_range

def cal_max_flat_range_simple(flat_img_width, resolution):
    '''
    calculate 1/2 flat img width in m
    flat_img_width in pixel
    resolution: refer to flat image resolution, due to geo_correct method, here, flat image resolution = slant image resolution, 
                    i.e., cal_max_flat_range_simple = cal_max_flat_range
    '''
    max_flat_range = flat_img_width/2 * resolution
    return max_flat_range


def cal_cmg(current_x, current_y, previous_x, previous_y):
    '''
    calculate adjencent ping cmg.
    current_x, current_y: fish spatial coords in meter
    previous_x, previous_y: the last ping fish spatial coords in meter
    return:
        cmg: degree(0-360,from north rotate clockwize to 航向)
    '''
    dx = current_x-previous_x
    dy = current_y-previous_y
    dx_abs = np.abs(dx)
    dy_abs = np.abs(dy)
    #quo = np.divide(dx, dy, out=np.zeros_like(dx), where=dy!=0)
    if dx>=0 and dy>0:
        cmg = np.arctan(np.divide(dx_abs, dy_abs))
    elif dx<=0 and dy<0:
        cmg = np.pi+np.arctan(np.divide(dx_abs, dy_abs))
    elif dx>0 and dy<0:
        cmg = np.pi-np.arctan(np.divide(dx_abs, dy_abs))
    elif dx<0 and dy>0:
        cmg = np.pi*2-np.arctan(np.divide(dx_abs, dy_abs))
    elif dx>0 and dy==0:
        cmg = np.pi/2
    elif dx<0 and dy==0:
        cmg = np.pi*1.5
    
    cmg = cmg*180./np.pi

    return cmg

def cal_fish_cmgs(x, y):
    '''
    x: fish spatial x coords in meter, shape=(ping_count,)
    y: fish spatial y coords in meter, shape=(ping_count,)
    return:
        cmgs: shape=(ping_count-1,)
    '''
    cmgs = []
    for i in range(0, x.shape[0]-1):
        cmg = cal_cmg(x[i+1], y[i+1], x[i], y[i])
        cmgs.append(cmg)

    return cmgs




def fill_gap(cur_ping_x, cur_ping_y, cur_ping_value, \
             pre_ping_x, pre_ping_y, pre_ping_value, \
             img, cmg, is_ns=False):
    '''
    fill gap between two pings.
    Note: 不能有相同的坐标, 且两帧坐标均需y不重叠,同一y多个x
    input: 
        first 6: 1-d ndarray, (flat sample count,)
        img: image to be filled
        cmg: current ping cmg, degree(0-360°)
        is_ns: survey line direction, if south-north, True; if west-east, False
    Return:
        filled image
    '''
    if is_ns:
        if cmg >= 90. and cmg <270.:
            pre = np.vstack((cur_ping_y,cur_ping_x,cur_ping_value)).T
            cur = np.vstack((pre_ping_y, pre_ping_x,pre_ping_value)).T
        else:
            cur = np.vstack((cur_ping_y,cur_ping_x,cur_ping_value)).T
            pre = np.vstack((pre_ping_y, pre_ping_x,pre_ping_value)).T
    else:
        if cmg >= 180.:
            cur = np.vstack((cur_ping_x,cur_ping_y,cur_ping_value)).T
            pre = np.vstack((pre_ping_x,pre_ping_y, pre_ping_value)).T
        else:
            cur = np.vstack((pre_ping_x,pre_ping_y, pre_ping_value)).T
            pre = np.vstack((cur_ping_x,cur_ping_y,cur_ping_value)).T

    cur = cur.astype(np.int64)
    pre = pre.astype(np.int64)
    y_min = max(np.min(cur[:,1]),np.min(pre[:,1]))
    y_max = min(np.max(cur[:,1]),np.max(pre[:,1]))
    for y in range(y_min, y_max+1):
        pre_x = pre[np.where(pre[:,1]==y)][:,0]
        cur_x = cur[np.where(cur[:,1]==y)][:,0]
        max_pre_x = np.max(pre_x) #针对一条线由多条并列竖线构成的情况
        max_cur_x =np.max(cur_x)
        if max_pre_x>max_cur_x: #pre 在cur右边
            min_pre_x = np.min(pre_x)
            if min_pre_x - max_cur_x >=2: #pre和cur 之间有缝隙
                fill_x = np.arange(max_cur_x+1, min_pre_x)
                fill_y = np.array([y] * fill_x.shape[0])
                neighbor_pre = pre[np.where((pre[:,0]==min_pre_x) & (pre[:,1]==y))][:,2]
                neighbor_cur = cur[np.where((cur[:,0]==max_cur_x) & (cur[:,1]==y))][:,2]
                
                if is_ns: #航线为南北时填充
                    if np.all(img[fill_x, fill_y]==0): #防止帧交叉严重时，缝隙填充失效情形
                        img[fill_x, fill_y]=(neighbor_cur+neighbor_pre)/2
                else: #航线为东西时填充
                    if np.all(img[fill_y, fill_x]==0): #防止帧交叉严重时，缝隙填充失效情形
                        img[fill_y, fill_x]=(neighbor_cur+neighbor_pre)/2 

def geocoding(flat_bs, proj_coords, coords_tck, coords_u, \
              slant_range_pixel: int, slant_range: float, \
              geocoding_img_res: float=0.1, is_ns: int=1, \
                fish_trajectory_smooth_type: str='bspline',\
                    angle_type = 'cog', angle=None, \
                        angle_tck=None, angle_u=None
                            ):
    #extrac tow fish projected coords
    fish_x_interpolate = proj_coords[:,0]
    fish_y_interpolate = proj_coords[:,1]
    tck_fun = coords_tck
    coef = coords_u

    # prepare for coord transfomation
    flat_res = slant_range/(slant_range_pixel/2) # due to the specific method of geo_correction, flat resolution = slant resolution at across survy line
    max_flat_range =   cal_max_flat_range_simple(flat_bs.shape[1], flat_res)
    ping_count = flat_bs.shape[0]
    flat_sample_count = flat_bs.shape[1]

    #geo_coding image parameters setting
    (img_geocoding_upperleft_x,img_geocoding_upperleft_y), (H,W) = \
        cal_geocodingimg_size(fish_x_interpolate,fish_y_interpolate, max_flat_range, geocoding_img_res)
    img_geocoding = np.zeros((H,W),dtype='uint8')
    
    fish_anlges = []
    fish_x_interping = []
    fish_y_interping = []
    
    prevping_geocoding_x = [] # for gap filling
    prevping_geocoding_y = []
    
    for i in range(0, ping_count):
        # determine interpolated ping counts
        if i == ping_count-1:
            inter_count = 1
            fish_x = [fish_x_interpolate[i]]
            fish_y = [fish_y_interpolate[i]]
            fish_x_interping += fish_x
            fish_y_interping += fish_y
            if angle_type == 'cog':
                angle_inter = cal_fish_cmgs(fish_x_interpolate[i-1:i+1],fish_y_interpolate[i-1:i+1])
            else:
                angle_inter = [angle[i]]

        else:
            dis = np.sqrt((fish_x_interpolate[i+1]-fish_x_interpolate[i])**2+(fish_y_interpolate[i+1]-fish_y_interpolate[i])**2)
            inter_count = int(np.ceil(dis/geocoding_img_res))
            if inter_count<=1:
                inter_count = 1
            #inter_count = 1
            if fish_trajectory_smooth_type == 'bspline':
                coef_current = coef[i]
                coef_next = coef[i+1]
                coef_inter = np.linspace(coef_current, coef_next, inter_count+1)
                fish_x, fish_y = interpolate.splev(coef_inter,tck_fun)
            if fish_trajectory_smooth_type == 'linear':
                fish_x = np.linspace(fish_x_interpolate[i], fish_x_interpolate[i+1], inter_count+1)
                fish_y = np.array([fish_y_interpolate[i]]*(inter_count+1))
            
            if angle_type == 'cog':
                angle_inter = cal_fish_cmgs(fish_x, fish_y)
            else:
                anlge_u_inter = np.linspace(angle_u[i], angle_u[i+1], inter_count)
                _, angle_inter = interpolate.splev(anlge_u_inter,angle_tck)
                angle_inter = angle_inter.tolist()

            fish_x_interping += (fish_x[:-1].tolist())
            fish_y_interping += (fish_y[:-1].tolist())

        img_resize = cv2.resize(flat_bs[i], (inter_count, flat_sample_count)).T
        fish_anlges += angle_inter

        #logging.info(f'i: {i}/{ping_count}, intercount: {inter_count}')

        # coords transformation, fusing, filling
        for j in range(0,inter_count):
            # flat image coords to spatial coords and calculation
            ping_towcoord_xt = pingcoord2towcoord(img_resize[j], max_flat_range)
            ping_towcoord_yt = np.full(flat_sample_count,0)
            ping_towcoord_zt = np.full(flat_sample_count,0)
            ping_towcoord = np.vstack((ping_towcoord_xt, ping_towcoord_yt, ping_towcoord_zt)).T

            fish_spatialcoord_reshape = np.array([fish_x[j], fish_y[j],0])
            fish_spatialcoord_reshape = fish_spatialcoord_reshape.reshape(1,fish_spatialcoord_reshape.shape[0])
            spatialcoord = towcoord2spatialcoord(ping_towcoord, (0.,0.,angle_inter[j]), fish_spatialcoord_reshape) #[x,y,z]

            #spatial coords to geocoding image coords
            spatial_x = spatialcoord[:,0]
            spatial_y = spatialcoord[:,1]
            imgcoord_x, imgcoord_y = \
                pingspatialcoord2imgcoord(spatial_x,spatial_y, img_geocoding_upperleft_x, img_geocoding_upperleft_y,geocoding_img_res)
            imgcoord_y_int=imgcoord_y.astype(np.int64)
            imgcoord_x_int=imgcoord_x.astype(np.int64)

            # remove duplicate coord and pixel for acceleration
            imgcoord = np.vstack((imgcoord_x_int, imgcoord_y_int)).T
            imgcoord_rm, index = prepp.remove_adjacent_duplicates(imgcoord)
            ori_ping_rm = flat_bs[i][index] 
            imgcoord_x_rm = imgcoord_rm[:,0]
            imgcoord_y_rm = imgcoord_rm[:,1]
                    
            
            # fuse,fill
            for id, coord in enumerate(imgcoord_rm):
                img_geocoding[coord[1],coord[0]] = ori_ping_rm[id] #新帧覆盖旧值
                # if img_geocoding[coord[1],coord[0]] == 0: #保留原有值,抛弃新帧值
                #     img_geocoding[coord[1],coord[0]] = ori_ping_rm[id] #丢弃新帧，保留旧值
                # else:
                #     img_geocoding[coord[1],coord[0]] = np.uint8((float(ori_ping_rm[id])+float(img_geocoding[coord[1],coord[0]]))/2.) #新旧值均值

            if i==0 and j==0:
                prevping_geocoding_x = imgcoord_x_rm
                prevping_geocoding_y = imgcoord_y_rm
                prev_ping = ori_ping_rm
            else:
                cmg = angle_inter[j]%(2*180.)
                fill_gap(imgcoord_x_rm, imgcoord_y_rm, ori_ping_rm, 
                            prevping_geocoding_x, prevping_geocoding_y, prev_ping, 
                            img_geocoding, cmg, is_ns)

                prevping_geocoding_x = imgcoord_x_rm #save previous ping info
                prevping_geocoding_y = imgcoord_y_rm
                prev_ping = ori_ping_rm
            
            # cv2.namedWindow('coding',0)
            # #cv2.resizeWindow('coding', (1000,800))
            # cv2.imshow('coding', img_geocoding)
            # cv2.waitKey(10)

    # save
    #cv2.imwrite(os.path.join(root, save_file, f'{name_index}-geocoding.png'), img_geocoding)
    #np.save(os.path.join(root, save_file, f'{name_index}-geocoding.npy'), img_geocoding)

    return (img_geocoding, 
            img_geocoding_upperleft_x, img_geocoding_upperleft_y, 
            np.array([fish_x_interping, fish_y_interping]).T, # tow fish projected coords after interpolating pings for gap filling
            fish_anlges) # ping directions after interpolating pings for gap filling            

def main():
    import argparse
    parser = argparse.ArgumentParser(description='detect sea bottom with gradient-mean-abnormal_detection-smooth')
    parser.add_argument('--root', default='D:/SidescanData_202312/jincheng_80/', help='root')
    parser.add_argument('--save_file', default='test', help='file name of saved file')
    parser.add_argument('--slant_range', default=80., type=float, help='slant_range')
    parser.add_argument('--geo_EPSG', default=4490, type=int, help='geographic coordinate system')
    parser.add_argument('--proj_EPSG', default=4499, type=int, help='projection coordinate system')
    parser.add_argument('--geocoding_img_res', default=0.1, type=float, help='geocoded image resolution')
    parser.add_argument('--is_ns', default=1, type=int, help='survey line direction, for fill')
    parser.add_argument('--fish_trajectory_smooth_type', default='bspline', type=str, help='smooth method')
    parser.add_argument('--fish_trajectory_smooth_factor', default=10., type=float, help='larger, smoother')
    parser.add_argument('--angle', default='cog', type=str, help='ping direction, cog, heading, course')
    parser.add_argument('--angle_smooth_factor', default=10., type=float, help='larger, smoother')
    args = parser.parse_args()

    root = args.root
    save_file = args.save_file
    slant_range = args.slant_range
    geocoding_img_res = args.geocoding_img_res
    is_ns = args.is_ns
    fish_trajectory_smooth_type = args.fish_trajectory_smooth_type
    fish_trajectory_smooth_factor = args.fish_trajectory_smooth_factor

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

    slant_bs_names = ['jincheng6_20-bs.npy']
    for n, slant_name in enumerate(slant_bs_names):
        # read variables
        name_index = slant_name.split('.')[-2] 
        slant_bs = np.load(os.path.join(root, 'npy', 'downsample',slant_name)) #(ping_count, sample_count)
        flat_bs = cv2.imread(os.path.join(root,'tile', 'gray_enhancement_coarse2fine',f'{name_index}-grayenhance.png'),0) #(ping_count, flat_sample_count)
        geo_coords = np.load(os.path.join(root, 'npy', 'raw',slant_name).replace('bs','geo_coords')) #(ping_count, 2), (lon,lat) with unit degree
        geo_coords = geo_coords[:,:2]
        if args.angle == 'heading':
            heading = np.load(os.path.join(root, 'npy', 'raw',slant_name).replace('bs','attitudes')) #(heading, pitch, roll, ...)
            angle = heading[:,0]
        if args.angle == 'course':
            course = np.load(os.path.join(root, 'npy', 'raw',slant_name).replace('bs','nmeas'))  #(spped, course)
            angle = course[:,1]
        if args.angle == 'cog':
            angle = None

        slant_bs = slant_bs[1252:2141,:]
        flat_bs = flat_bs[1252:2141,:]
        geo_coords = geo_coords[1252:2141,:]
        #angle = angle[:100]

        #preprocessing
        proj_coords, coords_tck, coords_u = \
            prepp.coords_prep(geo_coords, 
                              fish_trajectory_smooth_type, 
                                fish_trajectory_smooth_factor,
                                    args.geo_EPSG, args.proj_EPSG) #clean, geographic coordinates to projected coordinates, smooth; (coords_tck, coords_u): smooth parameters
        
        if angle is not None:
            angle, angle_tck, angle_u = prepp.sequence_prep(angle) #clean, smooth; (angle_tck, angle_u): smooth factor
        else:
            angle_tck = None 
            angle_u = None

        # geocoding
        logging.info(f'{n}/{len(slant_bs_names)}, {name_index}, i: {flat_bs.shape[0]}')
        slant_range_pixel = slant_bs.shape[1]
        geocoding_img, upperleft_x, upperleft_y, proj_interping_coords, angle_interping = \
            geocoding(flat_bs, 
                      proj_coords, coords_tck, coords_u, 
                      slant_range_pixel, 
                      slant_range, geocoding_img_res, is_ns, 
                      fish_trajectory_smooth_type, args.angle, angle, angle_tck, angle_u)
  
        # save geo tiff
        save_geotiff(geocoding_img, upperleft_x, upperleft_y, geocoding_img_res, 
                 args.geo_EPSG, args.proj_EPSG,
                os.path.join(root, 'tile', save_file, f'{name_index}-geocoding-cog1-0204.tif'))

if __name__=='__main__':
    main()
