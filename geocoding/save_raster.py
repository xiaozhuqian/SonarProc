from osgeo import gdal, osr

def array2raster(f_name, np_array, driver='GTiff',
                 prototype=None,
                 xsize=None, ysize=None,
                 transform=None, projection=None,
                 dtype=None, nodata=None):
    """
    https://theonegis.gitbook.io/geopy/gdal-kong-jian-shu-ju-chu-li/gdal-shu-ju-ji-ben-cao-zuo/zha-ge-shu-ju-chuang-jian-yu-bao-cun
    将ndarray数组写入到文件中
    :param f_name: 文件路径
    :param np_array: ndarray数组
    :param driver: 文件格式驱动
    :param prototype: 文件原型
    :param xsize: 图像的列数
    :param ysize: 图像的行数
    :param transform: GDAL中的空间转换六参数
    :param projection: 数据的投影信息
    :param dtype: 数据存储的类型
    :param nodata: NoData元数据
    """
    # 创建要写入的数据集（这里假设只有一个波段）
    # 分两种情况：一种给定了数据原型，一种没有给定，需要手动指定Transform和Projection
    driver = gdal.GetDriverByName(driver)
    if prototype:
        dataset = driver.CreateCopy(f_name, prototype)
    else:
        if dtype is None:
            dtype = gdal.GDT_Float32
        if xsize is None:
            xsize = np_array.shape[-1]  # 数组的列数
        if ysize is None:
            ysize = np_array.shape[-2]  # 数组的行数
        dataset = driver.Create(f_name, xsize, ysize, 1, dtype)  # 这里的1指的是一个波段
        dataset.SetGeoTransform(transform)
        dataset.SetProjection(projection)
    # 将array写入文件
    dataset.GetRasterBand(1).WriteArray(np_array)
    if nodata is not None:
        dataset.GetRasterBand(1).SetNoDataValue(nodata)
    dataset.FlushCache()
    return f_name

def save_geotiff(array, upperleft_x, upperleft_y, resolution, geo_EPSG, proj_EPSG, save_path):
    '''
    只适用于CGCS 2000地理坐标, Gauss-Kruger 6-degree 投影坐标,保存8位
    array: ndarray
    upperleft_x, upperleft_y: 左上角投影坐标/米
    resolution: map resolution, m/pixel
    zone: int 分带号
    save_path: 保存路径
    '''
    x_size = array.shape[1] # 图像列数
    y_size = array.shape[0] # 图像行数

    # https://blog.csdn.net/KilllerQueen/article/details/124843937
    srs = osr.SpatialReference()
    srs.SetProjCS( "CGCS2000 / Gauss-Kruger zone 20" )
    srs.ImportFromEPSG(geo_EPSG) # geographic coordinates CGCS2000
    srs.ImportFromEPSG(proj_EPSG) # projected coordinates CGCS2000/Gauss-Kruger zone20
    proj = srs.ExportToPrettyWkt()

    trans = (upperleft_x, resolution, 0.0, \
              upperleft_y, 0.0, -resolution)   #https://blog.csdn.net/RSstudent/article/details/108732571

    array2raster(save_path, array,
                xsize=x_size, ysize=y_size,
                transform=trans, projection=proj,
                dtype=gdal.GDT_Byte) #dtype meaning: https://blog.csdn.net/weixin_40625478/article/details/107839548


if __name__ == '__main__':
    import cv2
    upperleft_x_spatial = 20748568.444980677
    upperleft_y_spatial = 4137411.292008754
    geocoding_resolution = 0.1
    # upperleft_x_spatial=20748625.820210062
    # upperleft_y_spatial=4137416.786833383
    
    img = cv2.imread('D:/SidescanData_202312/deep_70/geocoding/deep1_20-bs-geocoding.png',
                     0)
    save_geotiff(img, upperleft_x_spatial, upperleft_y_spatial, geocoding_resolution, 
                 4490, 4498,
                      './deep1_1.tif')
    
   
