"""
2. Script to crop the raster stack (UAV images with multiple channels) with Chessboard Segments (shapefile with selected
samples exported from eCognition) and write sample patches to separate file each segment (30*30 pixel),

Note: raster file will be written without CRS
"""

from pathlib import Path
import geopandas as gpd
import shapely
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.transform import Affine

import logging
import time



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


#----- define the root

root = Path(r'M:\__crop_samples\stacks')
# r'M:\__crop_samples\stacks'

# place the shp folder inside the folder with stacks !
# -- M:\__crop_samples\stacks
# ------ CIR_RP_16_WR
# -------- CIR_RP_16_WR/shp



def extract_sample_patch_to_file(class_name, geometry, fid, indx, src):
    '''
    class_name: str
        exmp.: WR_2_0_2,
    geometry: str
        polygon as WKT
        POLYGON ((637886.073577 5645823.316707999, ... 637886.073577 5645823.316707999))
    src: <class 'rasterio.io.DatasetReader'>
        exmp.: src = rasterio.open(raster.as_posix())
    '''

    # create a folder to store patches according pattern name
    out_root = output_dir.joinpath(class_name)
    if not out_root.exists():
        out_root.mkdir(parents=True, exist_ok=True)
    out_tif = out_root.joinpath("{}_{}_{}.tif".format(fid, class_name, indx))

    # cut pattern patch according vector file
    coords = [shapely.wkt.loads(str(geometry))]
    # clipped_transform is not used, due to the CRS for CNN is not needed, therefore is ignored and 3d array simple
    # sliced along, x & y axis to 30 pixels
    clipped_array, clipped_transform = mask(dataset=src, shapes=coords, crop=True, all_touched=True, filled=False)
    # slice the 3D array along, x & y axis to 30 pixels
    clipped_array = clipped_array[:, :30, :30]

    logger.info('shape of the path array - clipped_array - {}'.format(clipped_array.shape))
    logger.info('clipped_transform): '.format(clipped_transform))
    # transform = Affine.translation(x[0] - xres / 2, y[0] - yres / 2) * Affine.scale(xres, yres)

    # set metadate weight and height to the size of the array
    # out_meta['width'] = clipped_array.shape[1]
    # out_meta['height'] = clipped_array.shape[2]
    xy_size = min([clipped_array.shape[1], clipped_array.shape[2]])
    out_meta['width'] = xy_size
    out_meta['height'] = xy_size
    logger.info('meta information: {}'.format(out_meta))

    # write cropped raster's -> crs ist probably wrong but we don't need it anyway for this task!
    with rasterio.open(out_tif, mode="w", **out_meta) as dst:
        dst.write(clipped_array)






ifile = Path(root)

for i in ifile.iterdir():
    if i.is_dir():
        tifs = i.glob("*{}*.tif".format('.v'))
        # find raster to be cropped
        for raster in tifs:
            # read header to variables
            sensor, b, c, crs, aoi, year, d, fid, date = Path(raster).stem.split('_')[1:10]
            a, b, c, d = None, None, None, None
            logger.info('sensor: {}, crs:{}, aoi:{}, year:{}, fid:{}, date:{}'.format(sensor, crs, aoi, year, fid, date))

            start_time = time.time()

            shps = i.glob("**/*{}*{}*.shp".format(fid, date))

            # find shapefile according to the date
            for shapefile in shps:
                logger.info('shapefile : {}'.format(shapefile))

                # crop the raster file
                # source: https://gist.github.com/mhweber/1af47ef361c3b20184455060945ac61b
                shp = gpd.read_file(shapefile.as_posix(), encoding='ANSI')

                src = rasterio.open(raster.as_posix())
                shp = shp.to_crs(src.crs)
                out_meta = src.meta.copy()

                output_dir = root.joinpath('samples_tiles\{}\{}'.format(sensor, aoi))
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)

                if len(shp.columns.tolist()[:-1]) == 1:
                    for indx, row in shp.iterrows():
                        class_name = row['Class_name']
                        geometry = row['geometry']
                        logger.info('geometry: {}'.format(geometry))

                        extract_sample_patch_to_file(class_name, geometry, fid, indx, src)
                else:
                    for col in shp.columns.tolist()[:-1]:
                        class_name = col
                        for indx, row in shp.loc[shp[col] == 1].iterrows():
                            geometry = row['geometry']

                            extract_sample_patch_to_file(class_name, geometry, fid, indx, src)


                end_time = time.time()
                hours, rem = divmod(end_time - start_time, 3600)
                minutes, seconds = divmod(rem, 60)
                print(" cropping of the pattern {} is finished: {:0>2}:{:0>2}:{:05.2f}".format(class_name, int(hours), int(minutes), seconds))




'''
clipped = rasterio.open(out_tif)
fig, ax = plt.subplots(figsize=(8, 6))
p1 = shp.plot(color=None,facecolor='none',edgecolor='red',linewidth = 2,ax=ax)
show(clipped, ax=ax)
ax.axis('off')
'''