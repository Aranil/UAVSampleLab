'''
1. Script to
    - crop UAV image after boundaries as WKT extracted from RCM.db with inner buffer
    - resample image to defined resolution (0.6)
    - calculate Color Spaces
    - write single bands to file
'''


import pandas as pd
import numpy as np
from pathlib import Path
from osgeo import gdal as gdal
from raster import Raster, multi_otsu_tresholding, combine_channels_to_multiband
from pyproj import Transformer
from pyproj import CRS

from dbflow.src.db_utility import connect2db
import config as cfg



def crop_uav_images():
    #os.environ['PROJ_LIB'] = datadir.get_data_dir()
    #print(os.environ['PROJ_LIB'])  # Confirm the path is correct

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    print(transformer.transform(0, 0))  # Should return transformed coordinates

    from pyproj import CRS
    crs = CRS.from_epsg(4326)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633")
    print(transformer.transform(0, 0))



    db_path=r'...\RCM.db'
    # connect to DB
    dbarchive = connect2db(db_path)
    root = r'...\_to_crop'
    s1root =r'..\xx_04_data\S1\GRD\bands'


    # THIS is the same as below but requires large memory volume for large images
    Transform_to_Color_Space = False
    Transform_to_Color_Space_chunked = False
    Extract_Bands_to_file = False
    Reproject_Raster = True


    # show all tables in DB
    # all_tables = dbarchive.engine.table_names()
    # all_tables

    # ----------------------------------------------------------------------------------------------------------------
    # I. Crop the UAV Images/DEM after Field Boundaries
    # ----------------------------------------------------------------------------------------------------------------

    # UAV_CIR_ortho_xx_32633_MRKN_2021_fid_A1270-20_20210601.tif - example of uav image header

    for FID in ['A1270-20']:
        sl_nummer = [FID]
        TABLE_NAME = 'areaofinterest'
        date = '20210601'

        # 'FRIEN' / 'MRKN' / 'DEMM'
        aoi = 'MRKN'

        # 'ortho' /'DEM'
        DATA_TYPE = 'ortho' # ortho image
        # DATA_TYPE = 'DEM' # digital elevation model

        # 'CIR' / 'RGB'
        SENSOR = 'CIR' # uav color infrared sensor
        # SENSOR = 'RGB' # uav with rgb sensor

        geometry_column = 'field_geom'
        buffer = cfg.shp_buffer_dist[aoi]


        # Define as projected Cartesian CRS not as  a geographic CRS (lat/lon, WGS84).
        # this Sentinel-1 Images are used as reference image to extract raster information
        # (resolution, crs, grid alignment etc.) to create a raster file
        if aoi == 'MRKN':
            CRS = '32633'
            reference_raster = f'{s1root}\MRKN\dB_32633\S1A__IW___A_20170113T165944_VH_NR_Orb_TF_TC.tif'
        elif aoi == 'FRIEN':
            CRS = '32632'
            reference_raster = f'{s1root}\FRIEN\dB_32632\S1B__IW___D_20201214T052513_VV_NR_Orb_TF_TC.tif'
        elif aoi == 'DEMM':
            CRS = '32633'
            reference_raster = f'{s1root}\DEMM\dB_32633\S1B__IW___D_20201221T051604_VV_NR_Orb_TF_TC.tif'

        if SENSOR == 'RGB':
            resolution = 0.03
        elif SENSOR == 'CIR':
            resolution = 0.06

        #from osgeo import gdal
        #print(gdal.GetConfigOption('PROJ_LIB'))
        #gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'EPSG')


        YEAR = date[0:4]

        # MAKE SURE THAT CRS OF SQL is equal to CRS of REFERENCE RASTER (SAR & NDVI)
        for SLNR in sl_nummer:
            sql = None
            sql = f"""SELECT
                         fid,
                         sl_nr,
                         aoi,
                         year,
                         crop_type_code,
                         AsText(ST_Buffer(ST_Transform("{geometry_column}", {CRS}), {buffer})) as geometry
                         FROM {TABLE_NAME}
                         WHERE
                         aoi="{aoi}"
                         AND year="{YEAR}"
                         AND sl_nr="{SLNR}"  ; """

        df = pd.read_sql(sql, dbarchive.archive.engine)
        print(df)

        # ------1. RESAMPLE UP the UAV image to make calculation faster
        resolution = 0.06

        if DATA_TYPE == 'ortho':
            ipath = r'{}\UAV_{}_{}_xx_{}_{}_{}_fid_{}_{}_{}.tif'.format(root, SENSOR, DATA_TYPE, CRS, aoi, YEAR, FID,
                                                                        date, str(resolution)[2:])
        elif DATA_TYPE == 'DEM':
            ipath = r'{}\UAV_{}_{}_xx_{}_{}_{}_fid_{}_{}.tif'.format(root, SENSOR, DATA_TYPE, CRS, aoi, YEAR, FID, date)


        if Path(ipath).exists():
            ifile = Raster(ipath)
            ofile_resamp1 = r'{}\UAV_{}_{}_xx_{}_{}_{}_fid_{}_{}_resamp{}.tif'.format(root, SENSOR, DATA_TYPE, CRS, aoi,
                                                                                      YEAR, FID.split('-')[0], date,
                                                                                      resolution)
            print(ofile_resamp1)
            print(CRS)
            print(ifile.proj)

            if not Path(ofile_resamp1).exists():

                resamp1_ds = ifile.resample_image(
                    ofile=ofile_resamp1,
                    ref_image=None,
                    to_epsg=CRS,
                    driver_format='GTiff',
                    cutlineDSName=db_path, # the DB that contains wkt
                    sql=sql,
                    crop2cutline=True,
                    xres=resolution,
                    yres=resolution,
                    bbox=None,
                    resample_alg=gdal.GRA_Bilinear, # use this for alignment the grid gdal.GRA_NearestNeighbour,
                    nan_value=np.nan
                )
                resamp1_ds = None




        if Transform_to_Color_Space_chunked:

            if Path(ofile_resamp1).exists():
                # Initialize Raster
                image_resamp = Raster(ofile_resamp1)

                # Select RGB channel combination
                selection = [3, 2, 1]
                chunk_size = 1024  # Set base chunk size

                # Transformation to different color spaces
                for space in ['LAB', 'LUV', 'YCbCr', 'HSI']:
                    print(f"Processing color space: {space}")
                    color_space = None
                    array_set = None

                    # Define channel names based on color space
                    if space == 'YCbCr':
                        color_space = ['Y', 'Cb', 'Cr']
                    elif space == 'HSI':
                        color_space = ['H', 'S', 'I']
                    elif space == 'LUV':
                        color_space = ['L', 'U', 'V']
                    elif space == 'LAB':
                        color_space = ['L', 'A', 'B']

                    # Store output filenames for each channel to combine them later
                    channel_files = []

                    # Process each channel in the color space
                    for ch in color_space:
                        # Define output filename for each channel
                        output_transform = r'{}\UAV_{}_ortho_xx_{}_{}_{}_fid_{}_{}_{}.tif'.format(
                            root, SENSOR, CRS, aoi, YEAR, FID.split('-')[0], date, ch
                        )
                        channel_files.append(output_transform)

                        # Create dataset for output
                        ds_transform = image_resamp.create_ds(
                            output_filename=output_transform,
                            bnumber=1,
                            fformat='GTiff',
                            datatype=gdal.GDT_Float32
                        )

                        # Process image in chunks
                        for i in range(0, image_resamp.ds.RasterYSize, chunk_size):
                            for j in range(0, image_resamp.ds.RasterXSize, chunk_size):
                                # Calculate actual width and height for the current chunk
                                width = min(chunk_size, image_resamp.ds.RasterXSize - j)
                                height = min(chunk_size, image_resamp.ds.RasterYSize - i)

                                # Read RGB blocks with adjusted chunk size at edges
                                block_R = image_resamp.ds.GetRasterBand(selection[0]).ReadAsArray(j, i, width, height)
                                block_G = image_resamp.ds.GetRasterBand(selection[1]).ReadAsArray(j, i, width, height)
                                block_B = image_resamp.ds.GetRasterBand(selection[2]).ReadAsArray(j, i, width, height)

                                # Apply the appropriate block function based on color space
                                if space == 'YCbCr':
                                    transformed_block = image_resamp.RGB2YCbCr_block(block_R, block_G, block_B)
                                elif space == 'HSI':
                                    transformed_block = image_resamp.RGB2HSI_block(block_R, block_G, block_B)
                                elif space == 'LUV':
                                    transformed_block = image_resamp.RGB2LUV_block(block_R, block_G, block_B)
                                elif space == 'LAB':
                                    transformed_block = image_resamp.RGB2LAB_block(block_R, block_G, block_B)

                                # Write the transformed block to the output file for the current channel
                                image_resamp.write_block(
                                    new_ds=ds_transform,
                                    array=transformed_block[ch],
                                    x_offset=j,
                                    y_offset=i,
                                    bnumber=1,
                                    nan=None
                                )
                        ch = None
                        # Close the dataset for the current channel
                        ds_transform = None
                        # ds_transform.close()
                        # image_resamp.close()

                    # Combine channels into a single multi-band file for the current color space
                    combined_filename = f"{root}/UAV_{SENSOR}_ortho_xx_{CRS}_{aoi}_{YEAR}_fid_{FID.split('-')[0]}_{date}_{space}.tif"
                    combine_channels_to_multiband(
                        output_filename=combined_filename,
                        channels=channel_files
                    )


        # THIS is the same as above but requires large memory volume for large images
        if Transform_to_Color_Space == True:

            if Path(ofile_resamp1).exists():
                # ------2. TRANSFORMATION to different Color Spaces
                image_resamp = Raster(ofile_resamp1)

                # select RGB channel combination
                selection = [3, 2, 1]

                # ------2.1 YCbCr
                # 'YCbCr', 'HSI', 'LUV'
                for space in ['LAB', 'HSI', 'LUV', 'YCbCr']:
                    print(space)
                    color_space = None
                    # select output
                    if space == 'YCbCr':
                        color_space = ['Y', 'Cb', 'Cr']
                        array_set = image_resamp.RGB2YCbCr(selection)
                    elif space == 'HSI':
                        color_space = ['H', 'S', 'I']
                        array_set = image_resamp.RGB2HSI(selection)
                    elif space == 'LUV':
                        color_space = ['L', 'U', 'V']
                        array_set = image_resamp.RGB2LUV(selection)
                    elif space == 'LAB':
                        color_space = ['A', 'B']
                        array_set = image_resamp.RGB2LAB(selection)

                    for ch in color_space:
                        output_transform = r'{}\UAV_{}_ortho_xx_{}_{}_{}_fid_{}_{}_{}.tif'.format(root, SENSOR, CRS,
                                                                                                  aoi, YEAR,
                                                                                                  FID.split('-')[0],
                                                                                                  date, ch)
                        ds_transform = image_resamp.create_ds(output_filename=output_transform,
                                                              bnumber=1,
                                                              fformat='GTiff',
                                                              datatype=gdal.GDT_Float32
                                                              )
                        image_resamp.write(
                                           new_ds=ds_transform,
                                           array=array_set[ch],
                                           bnumber=1,
                                           nan=None)
                        ds_transform = None


        if Extract_Bands_to_file == True:

            img = Raster(ofile_resamp1)
            for bn in range(1, img.bnumber + 1):
                # write channel to separate files
                ofile_band = r'{}\UAV_{}_ortho_xx_{}_{}_{}_fid_{}_{}_resamp{}_B{}.tif'.format(root, SENSOR, CRS, aoi,
                                                                                              YEAR,
                                                                                              FID.split('-')[0], date,
                                                                                              resolution, bn)
                img.extract_band(
                                 bname=bn,
                                 bnumber=[bn],
                                 ofile=ofile_band,
                                 func=None
                                 )
                ofile_band = None
            img = None




if __name__ == '__main__':
    crop_uav_images()

