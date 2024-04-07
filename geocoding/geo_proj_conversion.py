from osgeo import ogr
from osgeo import osr

#adapted from https://www.osgeo.cn/pygis/proj-osrproj.html

def geo2proj(lat, lon, geo_EPSG=4490, proj_EPSG=4498):
    '''
    lat/lon: degree, float
        4490-China Geodetic Coordinate System 2000
        4498 CGCS2000 / Gauss-Kruger zone 20
    return: [x,y] lat corresponds to y, lon corresponds to x
        if default EPSG, the first 2 bits of x are zone
            eg: 37.33093565914546, 119.8177438440776 is converted to
                [20749747.793371513, 4136964.2860312164]
    '''
    source = osr.SpatialReference()
    source.ImportFromEPSG(geo_EPSG) 

    target = osr.SpatialReference()
    target.ImportFromEPSG(proj_EPSG) 

    transform = osr.CoordinateTransformation(source, target) #source to target

    wkt = f'POINT ({lat} {lon})'   
    point = ogr.CreateGeometryFromWkt(wkt) #"POINT (lat lon)" lat/lon: degree
    
    point.Transform(transform)
    x = point.GetY()
    y = point.GetX()

    return [x,y]

def proj2geo(x, y, geo_EPSG=4490, proj_EPSG=4498):
    source = osr.SpatialReference()
    source.ImportFromEPSG(proj_EPSG) 

    target = osr.SpatialReference()
    target.ImportFromEPSG(geo_EPSG) 

    transform = osr.CoordinateTransformation(source, target)

    wkt = f'POINT ({y} {x})'  
    point = ogr.CreateGeometryFromWkt(wkt) #"POINT (4136964.286031217 20749747.793371513)" (y,x) the first 2 bits of x are zone
    point.Transform(transform)
    lat = point.GetX()
    lon = point.GetY()
    
    return [lat,lon]

if __name__ == "__main__":
    coord1 = geo2proj(37.33093565914546, 119.8177438440776, geo_EPSG=4490, proj_EPSG=4498)
    print(coord1)
    coord2 = proj2geo(20749747.793371513, 4136964.286031217, geo_EPSG=4490, proj_EPSG=4498)
    print(coord2)