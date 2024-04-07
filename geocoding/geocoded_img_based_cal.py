import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from scipy import interpolate
from coords_transform_based_cal import *

#https://stackoverflow.com/questions/16028752/how-do-i-get-all-the-points-between-two-point-objects
def bresenham_algorithm(v1, v2):
    vResult, incYi, incXi, incYr, incXr =[], 0, 0, 0, 0
    dY = (v2[1] - v1[1])
    dX = (v2[0] - v1[0])
    #Incrementos para secciones con avance inclinado
    if(dY >= 0):
        incYi = 1
    else:
        dY = -dY
        incYi = -1
        
    if(dX >= 0):
        incXi = 1
    else:
        dX = -dX
        incXi = -1
    #Incrementos para secciones con avance recto   
    if(dX >= dY):
        incYr = 0
        incXr = incXi
    else:
        incXr = 0
        incYr = incYi
        #Cuando dy es mayor que dx, se intercambian, para reutilizar el mismo bucle. Intercambio rapido de variables en python
        #k = dX, dX = dY, dY = k
        dX, dY = dY, dX
        
    # Inicializar valores (y de error).
    x, y = v1[0], v1[1]
    avR = (2 * dY)
    av = (avR - dX)
    avI = (av - dX)
    vResult.append([x,y]) #podemos pintar directamente
    while x != v2[0]:
        if(av >= 0):
            x = (x + incXi)
            y = (y + incYi)
            av = (av + avI)
        else:
            x = (x + incXr)
            y = (y + incYr)
            av = (av + avR)
        vResult.append([x,y]) #podemos pintar directamente
    vResult = np.array(vResult)
    
    return vResult

def cal_in_geocoding_img(j, fish_x, fish_y,cmgs_inter,img_resize,\
                     max_flat_range, geocoding_img_res,\
                        img_geocoding_upperleft_x, img_geocoding_upperleft_y):
    '''
    在编码图像坐标系中计算声呐坐标,西班牙硕士论文方法,以ping为单位
    j: ping
    fish_x,fish_y,cmgs_inter,img_resize: 当前帧插帧后的拖鱼坐标,cmg和bs
    img_geocoding_upperleft_x, img_geocoding_upperleft_y: geocoded imaga upperleft spatial coords in meter
    '''
    fish_spatialcoord_reshape = np.array([fish_x[j], fish_y[j],0])
    fish_spatialcoord_reshape = fish_spatialcoord_reshape.reshape(1,fish_spatialcoord_reshape.shape[0])
    fish_imgcoord_x, fish_imgcoord_y = \
        pingspatialcoord2imgcoord(fish_spatialcoord_reshape[:,0],fish_spatialcoord_reshape[:,1], img_geocoding_upperleft_x, img_geocoding_upperleft_y,geocoding_img_res)
    fish_imgcoord_x = int(fish_imgcoord_x)
    fish_imgcoord_y = int(fish_imgcoord_y)

    port_end_x = int(fish_imgcoord_x-max_flat_range/geocoding_img_res * np.cos(cmgs_inter[j]* (np.pi/180.)))
    port_end_y = int(fish_imgcoord_y-max_flat_range/geocoding_img_res * np.sin(cmgs_inter[j]* (np.pi/180.)))
    starboard_end_x = int(fish_imgcoord_x+max_flat_range/geocoding_img_res * np.cos(cmgs_inter[j]* (np.pi/180.)))
    starboard_end_y = int(fish_imgcoord_y+max_flat_range/geocoding_img_res * np.sin(cmgs_inter[j]* (np.pi/180.)))

    port_coords = bresenham_algorithm((fish_imgcoord_x, fish_imgcoord_y), (port_end_x, port_end_y))
    port_coords = np.array(port_coords.tolist()[::-1])[:-1,:]
    starboard_coords = bresenham_algorithm((fish_imgcoord_x, fish_imgcoord_y), (starboard_end_x, starboard_end_y))
    imgcoord_rm = np.vstack((port_coords, starboard_coords))

    imgcoord_x_rm = imgcoord_rm[:, 0]
    imgcoord_y_rm = imgcoord_rm[:, 1]

    flat_ping = img_resize[j]
    ori_ping_rm = cv2.resize(flat_ping, (1, imgcoord_rm.shape[0])).squeeze()

    return imgcoord_x_rm, imgcoord_y_rm, imgcoord_rm, ori_ping_rm

def main():
    root = 'D:/SidescanData_202312/deep_70'
    save_file = 'geocoding'
    slant_range = 70
    geocoding_img_res = 0.1
    fish_trajectory_smooth_type = 'bspline'
    fish_trajectory_smooth_factor = 60
    is_ns = True

    if not os.path.exists(os.path.join(root, save_file)):
        os.makedirs(os.path.join(root, save_file)) 

    slant_bs_names = []
    for slant_bs_path in sorted(glob.glob(os.path.join(root,'npy','downsample', '*-bs.npy'))):
        _, slant_bs_name = os.path.split(slant_bs_path)
        slant_bs_names.append(slant_bs_name)

    slant_bs_names = ['deep2_20-bs.npy']

    for n, slant_name in enumerate(slant_bs_names):
        # read variables
        name_index = slant_name.split('.')[-2]
        slant_bs = np.load(os.path.join(root, 'npy', 'downsample',slant_name)) #(ping_count, sample_count)
        flat_bs = np.load(os.path.join(root,'gray_enhancement_retinex',f'{name_index}-grayenhance.npy')) #(ping_count, flat_sample_count)
        geo_coords = np.load(os.path.join(root, 'npy', 'downsample',slant_name).replace('bs','geo_coords')) #(ping_count, 2), (lon,lat)
        geo_coords = geo_coords[:,:2]

        #fish geograypy coords clear, projection, interpolate
        geo_coords = geo_coords_clear(geo_coords) # fill the missing geo coords
        fish_spatialcoords = fish_geocoords2fish_spatialcoords(geo_coords) # fish geo coords projection
        if fish_trajectory_smooth_type == 'linear':
            fish_x_interpolate, fish_y_interpolate = fish_interpolate(fish_spatialcoords, 'linear')
        if fish_trajectory_smooth_type == 'bspline':
            fish_x_interpolate, fish_y_interpolate, tck_fun, coef = fish_interpolate(fish_spatialcoords, 'bspline', fish_trajectory_smooth_factor)
        
        # prepare for coord transfomation
        flat_res = slant_range/(slant_bs.shape[1]/2) # due to the specific method of geo_correction, flat resolution = slant resolution at across survy line
        max_flat_range =   cal_max_flat_range_simple(flat_bs.shape[1], flat_res)
        ping_count = flat_bs.shape[0]
        flat_sample_count = flat_bs.shape[1]

        #geo_coding image parameters setting
        (img_geocoding_upperleft_x,img_geocoding_upperleft_y), (H,W) = \
            cal_geocodingimg_size(fish_x_interpolate,fish_y_interpolate, max_flat_range, geocoding_img_res)
        img_geocoding = np.zeros((H,W),dtype='uint8')
        
        fish_cmgs = []
        fish_x_interping = []
        fish_y_interping = []
        
        prevping_geocoding_x = [] # for gap filling
        prevping_geocoding_y = []
        
        for i in range(0, 2000):
            # determine interpolated ping counts
            if i == ping_count-1:
                inter_count = 1
                fish_x = [fish_x_interpolate[i]]
                fish_y = [fish_y_interpolate[i]]
                fish_x_interping += fish_x
                fish_y_interping += fish_y
                cmgs_inter = cal_fish_cmgs(fish_x_interpolate[i-1:i+1],fish_y_interpolate[i-1:i+1])
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
                
                cmgs_inter = cal_fish_cmgs(fish_x, fish_y)
                fish_x_interping += (fish_x[:-1].tolist())
                fish_y_interping += (fish_y[:-1].tolist())

            img_resize = cv2.resize(flat_bs[i], (inter_count, flat_sample_count)).T
            fish_cmgs += cmgs_inter
            
            print(f'{n}/{len(slant_bs_names)}, {name_index}, i: {i}/{ping_count}, intercount: {inter_count}')
            #logger.info(f'{n}/{len(img_names)}, {name}, i: {i}/{ping_count}, j: {j}/{inter_count}')

            # coords transformation, fusing, filling
            for j in range(0,inter_count):
                imgcoord_x_rm, imgcoord_y_rm, imgcoord_rm, ori_ping_rm = cal_in_geocoding_img(\
                    j, fish_x, fish_y,cmgs_inter,img_resize,\
                     max_flat_range, geocoding_img_res,\
                        img_geocoding_upperleft_x, img_geocoding_upperleft_y)
   
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
                    cmg = cmgs_inter[j]%(2*180.)
                    fill_gap(imgcoord_x_rm, imgcoord_y_rm, ori_ping_rm, 
                                prevping_geocoding_x, prevping_geocoding_y, prev_ping, 
                                img_geocoding, cmg, is_ns)

                    prevping_geocoding_x = imgcoord_x_rm #save previous ping info
                    prevping_geocoding_y = imgcoord_y_rm
                    prev_ping = ori_ping_rm
                
                cv2.namedWindow(f'{name_index}-coding',0)
                #cv2.resizeWindow('coding', (1000,800))
                cv2.imshow(f'{name_index}-coding', img_geocoding)
                cv2.waitKey(10)

        # save
        cv2.imwrite(os.path.join(root, save_file, f'{name_index}-geocoding.png'), img_geocoding)
        #np.save(os.path.join(root, save_file, f'{name_index}-geocoding.npy'), img_geocoding)

        zone = int(str(img_geocoding_upperleft_x)[:2])
        save_geotiff(img_geocoding, img_geocoding_upperleft_x, img_geocoding_upperleft_y,\
                     geocoding_img_res, zone, os.path.join(root, save_file, f'{name_index}-geocoding.tif'))

    
if __name__=='__main__':
    main()
