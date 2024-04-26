'''stitch geocoding tiles to one raster data based on geographical coordinates;
    merge overlappling with maximum value'''
from osgeo import gdal
import os
import glob
import math
import numpy as np
import time

# refer to https://blog.51cto.com/u_15688229/5640065
# https://blog.csdn.net/m0_56180742/article/details/119696803
#获取影像的左上角和右下角坐标
def get_spatialcoords(ds):
    geotrans=list(ds.GetGeoTransform())
    xsize=ds.RasterXSize
    ysize=ds.RasterYSize
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]  
    return min_x,max_y,max_x,min_y

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stitch tile mosaics based on geographical coordinates')
    parser.add_argument('--root', default='./outputs/', help='root')
    parser.add_argument('--out_file_name', default='mosaic', help='file for output mosaic')
    
    args = parser.parse_args()
    root = args.root

    if not os.path.exists(os.path.join(root,  args.out_file_name)):
        os.makedirs(os.path.join(root, args.out_file_name)) 

    start =time.time()

    input_names = []
    for input_path in sorted(glob.glob(os.path.join(root,'tile','geocoding', '*20-bs-geocoding.tif'))):
        _, input_name = os.path.split(input_path)
        input_names.append(input_name)

    # read input geo tiff
    in_dss = {}
    spatial_coords = []
    for name in input_names:
        index = name.split('.')[-2]
        in_ds = gdal.Open(os.path.join(root,'tile','geocoding',name))
        in_dss[index] = in_ds
        min_x,max_y,max_x,min_y = get_spatialcoords(in_ds)
        spatial_coords.append([min_x,max_y,max_x,min_y])
    del in_ds

    in_ds0 = list(in_dss.values())[0]
    in_ds_trans = list(in_ds0.GetGeoTransform())
    in_ds_proj = in_ds0.GetProjection()
    in_ds_dtype = in_ds0.GetRasterBand(1).DataType
    spatial_coords = np.array(spatial_coords)
    del in_ds0

    # creat mosaic raster
    min_x = min(spatial_coords[:,0])
    max_y = max(spatial_coords[:,1])
    max_x = max(spatial_coords[:,2])
    min_y = min(spatial_coords[:,3])
    w_res = in_ds_trans[1]
    h_res = in_ds_trans[5]
    out_ds_columns = math.ceil((max_x-min_x)/w_res)
    out_ds_rows = math.ceil((max_y-min_y)/(-h_res))
    out_ds_dtype = in_ds_dtype
    out_ds_trans = (min_x, w_res, 0.0, max_y, 0.0, h_res)
    out_ds_proj = in_ds_proj

    mosaic_name = root.split('/')[-2] + '.tif'
    mosaic_path = os.path.join(root, args.out_file_name, mosaic_name)
    driver=gdal.GetDriverByName('GTiff')
    out_ds=driver.Create(mosaic_path,out_ds_columns,out_ds_rows,1,out_ds_dtype)
    out_ds.SetProjection(out_ds_proj)
    out_ds.SetGeoTransform(out_ds_trans)

    # write input rasters to output
    out_data = np.zeros((out_ds_rows, out_ds_columns))
    for name, in_ds in in_dss.items():
        trans=gdal.Transformer(in_ds,out_ds,[])#pixel coords transformation from one raster to another    
        _, xyz=trans.TransformPoint(False,0,0)#计算in_ds中左上角像元对应out_ds中的行列号    
        x,y,z=map(int,xyz)
        #print(x,y,z)
        in_data=in_ds.GetRasterBand(1).ReadAsArray()
        in_h, in_w = in_data.shape
        cur_data = np.zeros((out_ds_rows, out_ds_columns))
        cur_data[y:(y+in_h),x:(x+in_w)] = in_data
        out_data = np.maximum(out_data, cur_data) # merge in overlappling using maximum value

        in_ds.FlushCache() # clear cache

    out_ds.GetRasterBand(1).WriteArray(out_data)

    del in_ds, out_ds # close raster

    print('time cost:', time.time()-start)
    
