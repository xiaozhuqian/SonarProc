import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import scipy.signal
import peakutils.peak
from scipy import interpolate
import copy
import cv2


def peak_detection(coords_port, n_sample, peak_factor=0.2):
    '''
    coords_half: port bottom line coords, (port_x, y)
    n_sample: width of waterfall map
    peak_factor: ratio of peak depth to mean average depth 
    return:
        peak_coords: port peak coords, (port_peak_x, y)
    '''

    avg_y_port = np.mean(coords_port[:,0])
    avg_dis_port = int(n_sample/2 - avg_y_port) #distance between water column middle line and averaged bottom line points
    peak_height = int(n_sample/2 - avg_dis_port * peak_factor)
    px, _ = scipy.signal.find_peaks(coords_port[:,0], height=peak_height, distance = 50, prominence=20)
    peak_coords = np.array(list(zip(coords_port[px, 0], px)))

    return peak_coords

def valley_detection(coords_half, peak_coord, valley_std=5):
    '''
    usage:
        detect valley of one peak
    input:
        coords_half: port/starboard bottom line coords, (port_x/starboard_x, y)
        peak_coord: port/starboard peak coords, [port_peak_x/starboard_peak_x, y]
    return:
        valley_left_point: [x,y]
        valley_right_point: [x,y]
    '''
    coords_half = coords_half[:,[1,0]]
    peak_coord = [peak_coord[1], peak_coord[0]]
    
    coords_candidate = coords_half[max(peak_coord[0]-300, 0):min(peak_coord[0]+300, coords_half.shape[0])] #峰值左右各300窗口内，检测波谷
    vx_index = peakutils.peak.indexes(-coords_candidate[:,1],min_dist=5) #波谷坐标index检测
    vx_ori = coords_candidate[vx_index,0] #波谷x
    vx_left = vx_ori[vx_ori<peak_coord[0]][::-1] # port valley x
    vx_right = vx_ori[vx_ori>peak_coord[0]] # starboard valley x
    # select valley left and right point 
    for j in range(0,len(vx_right)-1):
        y_interval = coords_half[vx_right[j]:vx_right[j+1], 1] #coords between adjencent valley points
        std = np.std(y_interval)
        if std < valley_std:
            #vx_target_right = vx_right[j+3]
            vx_target_right = vx_right[j+2]
            vy_target_right = coords_half[vx_target_right,1]
            break
    for k in range(0,len(vx_left)-1):
        y_interval = coords_half[vx_left[k+1]:vx_left[k], 1]
        std = np.std(y_interval)
        if std < valley_std:
            vx_target_left = vx_left[k+2]
            vy_target_left = coords_half[vx_target_left,1]
            break
    
    return [vy_target_left, vx_target_left],[vy_target_right, vx_target_right]
    
def interpolate_line(coords, n_sample, valley_left_point, valley_right_point, interpolate_kind = 'linear'):
    '''
    usage:
        interpolate one valley
    input:
        coords: (x,y), n_ping*n_sample
        valley_left_point: [x,y]
        valley_right_point: [x,y]
    return:
        coords_inter: (x,y), n_ping*n_sample
    '''
    
    coords_port = coords[:, [2,0]]
    coords_starboard = coords[:, [2,1]]

    coords_port_inter = copy.deepcopy(coords_port)
    coords_starboard_inter = copy.deepcopy(coords_starboard)

    vx_target_left = valley_left_point[1]
    vx_target_right = valley_right_point[1]

    coords_sample = np.concatenate([coords_port[vx_target_left-10: vx_target_left+1],\
                                             coords_port[vx_target_right: vx_target_right+10]],axis=0) #波谷左右两点分别向外延伸10个点作为差值方程求取样本点
    x_interpolate = coords_sample[:,0]
    y_interpolate = coords_sample[:,1]
    f=interpolate.interp1d(x_interpolate,y_interpolate,kind=interpolate_kind) #差值方程
    #在波谷左右两点内插值
    x_new = coords_port[vx_target_left+1: vx_target_right, 0] 
    y_new = f(x_new).astype(int)
    
    #用插值点替代原异常点
    coords_port_inter[vx_target_left+1: vx_target_right, 1] = y_new
    coords_starboard_inter[vx_target_left+1: vx_target_right, 1] = n_sample-y_new #对称替换右舷

    coords_inter = [[x[1],y[1],y[0]] for x, y in zip(coords_port_inter,coords_starboard_inter)]
    coords_inter = np.array(coords_inter, dtype=int)
    
    return coords_inter

def ab_line_correction(coords, n_sample, peak_factor, valley_std):
    coords_port = coords[:, [0,2]]
    peak_coords = peak_detection(coords_port, n_sample, peak_factor)
    coords_inter = copy.deepcopy(coords)
    for peak in peak_coords:
        valley_left, valley_right = valley_detection(coords_port, peak, valley_std)
        coords_inter = interpolate_line(coords_inter, n_sample, valley_left, valley_right)

    return coords_inter

if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from utils.vis import gray_enhancing_tvg, npy2img, draw_points_on_image
    
    coords_dir = 'D:/SidescanData_202312/sanshan_80/bottom_line/'
    bs_dir = 'D:/SidescanData_202312/sanshan_80/npy/downsample/'
    out_dir = 'D:/SidescanData_202312/sanshan_80/bottom_line/interpolate/'
    peak_factor = 0.25
    valley_std = 5

    bs_names = []
    for bs_path in sorted(glob.glob(os.path.join(bs_dir, '*.npy'))):
        _, bs_name = os.path.split(bs_path)
        bs_names.append(bs_name)

    bs_names = ['sanshan3_20-bs.npy']

    for bs_name in bs_names:
        index = bs_name.split('.')[-2]
        bs=np.load(os.path.join(bs_dir,  bs_name))
        n_sample = bs.shape[1]
        print(bs_name)
        
        coords_path = os.path.join(coords_dir, f'{index}-bottom_3smooth.txt')
        coords_port = []
        coords_starboard = []
        with open(coords_path) as f:
            for line in f: # transfer x and y for peak detection
                y_left = int(line.strip().split(' ')[0])
                y_right = int(line.strip().split(' ')[1])
                x = int(line.strip().split(' ')[2])
                coords_port.append([y_left,x])
                coords_starboard.append([y_right,x])
        coords_port = np.array(coords_port)
        coords_starboard = np.array(coords_starboard)
        coords = [[x[0],y[0],y[1]] for x, y in zip(coords_port,coords_starboard)]
        coords = np.array(coords, dtype=int)
    
        #coords_inter = ab_line_correction(coords, n_sample, peak_factor, valley_std)
        
        #abnormal line correction
        peaks = peak_detection(coords_port, n_sample, peak_factor)
        
        coords_inter = copy.deepcopy(coords)
        valley_lefts = []
        valley_rights = []
        for peak in peaks:
            valley_left, valley_right = valley_detection(coords_port, peak, valley_std)
            valley_lefts.append(valley_left)
            valley_rights.append(valley_right)
            coords_inter = interpolate_line(coords_inter, n_sample, valley_left, valley_right)

        # visualize original and interpolated bottom line
        coords_port_inter = coords_inter[:,[0, 2]]
        px = peaks[:,1]
        valley_left_xs = np.array(valley_lefts)[:,1]
        valley_right_xs = np.array(valley_rights)[:,1]
        plt.figure(figsize=(20,6))
        plt.subplot(211) #原海底线标注检测到的波峰
        plt.plot(coords_port[:,1], coords_port[:,0])
        plt.scatter(px, coords_port[px, 0], color="green",s=8)
        plt.subplot(212) #插值后的海底线
        plt.plot(coords_port_inter[:,1], coords_port_inter[:,0])
        plt.scatter(px, coords_port_inter[px, 0], color="green",s=8)
        plt.scatter(valley_left_xs, coords_port_inter[valley_left_xs, 0], color="red",s=8)
        plt.scatter(valley_right_xs, coords_port_inter[valley_right_xs, 0], color="red",s=8)
        plt.savefig(os.path.join(out_dir, f'{index}_inter_coords-plot.png'))
        #plt.show()

        #draw bottom line on image
        bs_tvg = gray_enhancing_tvg(bs, 70., 5.0, 0.05)
        img = npy2img(bs_tvg)
        port_coord = coords_inter[:, [0,2]].tolist()
        starboard_coord = coords_inter[:, [1,2]].tolist()
        points = port_coord + starboard_coord
        color_img = draw_points_on_image(points, img)

        #save
        cv2.imwrite(os.path.join(out_dir, f'{index}-bottom_inter.png'), color_img)
        np.savetxt(os.path.join(out_dir, f'{index}-bottom_inter.txt'), coords_inter, delimiter=' ', fmt='%d')







