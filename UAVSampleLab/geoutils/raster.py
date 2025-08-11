"""
@author: Aranil

Utils to handle high-resolution UAV data
- read
- functions to generate color spaces
"""

import numpy as np
from pathlib import Path

from osgeo import gdal as gdal
from osgeo import osr as osr
from skimage import exposure
import cv2
import copy
#from spatialist.envi import HDRobject

import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from skimage import color, morphology, segmentation
from skimage.measure import label, regionprops
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import seaborn as sns



def increase_brightness(img, value=30):
    '''
    source: https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    img[np.isnan(img)] = 255
    return img




class Raster:

    def __init__(self, filename='', xsize=0, ysize=0, bnumber=0):
        if filename:
            self._load_from_file(filename)
        elif xsize > 0 and ysize > 0 and bnumber > 0:
            self._create_empty_raster(xsize, ysize, bnumber)
        else:
            raise ValueError("Provide either a valid filename or dimensions (xsize, ysize, bnumber).")


    def __repr__(self):
        return f"<Raster: {self.filename or 'in-memory'}, size=({self.xsize}, {self.ysize}), bands={self.bnumber}, epsg={self.epsg}>"

    def __str__(self):
        return f"Raster from {self.filename or 'memory'} with {self.bnumber} band(s), dimensions: {self.xsize}x{self.ysize}"



    def _create_empty_raster(self, xsize, ysize, bnumber):
        self.filename = ''
        self.ysize= ysize
        self.xsize = xsize
        self.bnumber = bnumber
        self.array = np.zeros((ysize, xsize, bnumber), dtype=np.float32)
        self.stack_array = {i + 1: self.array[:, :, i] for i in range(bnumber)}

        # Placeholder attributes for consistency
        self.gtransform = None
        self.bbox = None
        self.wkt = None
        self.proj = None
        self.epsg = None
        self.nan = np.nan
        self.dtype = None
        self.dtype_as_str = None
        self.bname = None


    def _load_from_file(self, filename):

        self.path = Path(filename)

        if not self.path.exists():
            raise FileNotFoundError(f"File does not exist: {filename}")

        file_format = self._identify_format(self.path)
        self.fformat = file_format

        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        if ds is None:
            raise IOError(f"Could not open file: {filename}")
        self.ds = ds

        self.filename = ds.GetDescription()
        print('self.filename:', self.filename)
        self.gtransform = ds.GetGeoTransform()

        self.xsize = ds.RasterXSize
        self.ysize = ds.RasterYSize
        self.bnumber = ds.RasterCount

        self.minx = self.gtransform[0]
        self.maxy = self.gtransform[3]
        self.xres = self.gtransform[1]
        self.yres = self.gtransform[5]
        self.maxx = self.minx + self.xsize * self.xres
        self.miny = self.maxy + self.ysize * self.yres

        self.bbox = [self.minx, self.miny, self.maxx, self.maxy]
        self.wkt = (
            f"POLYGON (({self.minx} {self.miny}, {self.minx} {self.maxy}, "
            f"{self.maxx} {self.maxy}, {self.maxx} {self.miny}, {self.minx} {self.miny}))"
        )

        self.nan = ds.GetRasterBand(1).GetNoDataValue() or np.nan
        self.dtype = ds.GetRasterBand(1).DataType
        self.dtype_as_str = self._map_gdal_dtype(self.dtype)

        proj_wkt = ds.GetProjection()
        self.proj = osr.SpatialReference(wkt=proj_wkt)
        self.epsg = self.proj.GetAttrValue('AUTHORITY', 1) if self.proj.IsProjected() else None

        self.array = np.stack(
            [ds.GetRasterBand(i).ReadAsArray() for i in range(1, self.bnumber + 1)],
            axis=-1
        )

        self.bname = [ds.GetRasterBand(i).GetDescription() for i in range(1, self.bnumber + 1)]
        #self.blist = ds.GetFileList()


    def _identify_format(self, path_obj):
        ext = path_obj.suffix.lower()
        format_map = {
            '.tif': 'GTiff',
            '.vrt': 'VRT',
            '.nc': 'netCDF',
            '': 'ENVI',  # likely needs custom handling
        }
        if ext in format_map:
            return format_map[ext]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


    def _map_gdal_dtype(self, gdal_type):
        gdal_dtype = {
            1: "int8",
            2: "uint16",
            3: "int16",
            4: "uint32",
            5: "int32",
            6: "float32",
            7: "float64",
            10: "complex64",
            11: "complex128",
        }
        return gdal_dtype.get(gdal_type, "unknown")



    def create(self, output_filename, array=None, xsize=None, ysize=None, bnumber=None, bname=None,
               proj=None, gtransform=None, func=None, fformat=None, nan=None, datatype=gdal.GDT_Float64):
        """
        Creates a new raster file using current raster attributes or provided parameters.

        Parameters
        ----------
        output_filename : str
            Path to the output file.
        array : np.ndarray, optional
            2D array to write into each band.
        xsize, ysize : int, optional
            Dimensions of the raster.
        bnumber : int, optional
            Number of bands.
        bname : list of str, optional
            Names for each band.
        proj : str, optional
            WKT projection string.
        gtransform : tuple, optional
            GeoTransform (affine transform) tuple.
        func : callable, optional
            A function to apply to the array values before writing.
        fformat : str, optional
            GDAL driver format (e.g., 'GTiff', 'ENVI').
        nan : float, optional
            NoData value.
        datatype : GDAL data type, optional
            GDAL constant for data type (default: gdal.GDT_Float64).

        Returns
        -------
        Raster
            A new Raster object representing the created file.
        """

        # Determine output format
        driver_format = fformat or self.fformat or 'GTiff'
        driver = gdal.GetDriverByName(driver_format)

        # Fallback to instance attributes if not provided
        xsize = xsize or self.xsize
        ysize = ysize or self.ysize
        bnumber = bnumber or self.bnumber
        proj = proj or (self.proj.ExportToWkt() if self.proj else '')
        gtransform = gtransform or self.gtransform
        nan_value = nan if nan is not None else self.nan

        if not (xsize and ysize and bnumber and proj and gtransform):
            raise ValueError("Insufficient data to create raster. Check required attributes.")

        # Create dataset
        new_ds = driver.Create(output_filename, xsize, ysize, bnumber, datatype)
        new_ds.SetProjection(proj)
        new_ds.SetGeoTransform(gtransform)

        # Write each band
        for b in range(1, bnumber + 1):
            # Determine which array to use
            if array is not None:
                if isinstance(array, np.ndarray) and len(array.shape) == 2:
                    band_array = array
                else:
                    raise ValueError("Provided array must be a 2D numpy array.")
            else:
                band_array = self.ds.GetRasterBand(b).ReadAsArray().astype(np.float64)

            # Apply function if provided
            if func is not None:
                if nan_value == 0:
                    result = np.where(band_array > 0.0, func(band_array), np.nan)
                else:
                    result = func(band_array)
            else:
                result = band_array

            band = new_ds.GetRasterBand(b)
            band.WriteArray(result)
            band.SetNoDataValue(nan_value)

            if bname is not None:
                if not isinstance(bname, list):
                    raise TypeError("bname must be a list of strings.")
                if len(bname) != bnumber:
                    raise ValueError(f"bname length ({len(bname)}) must match number of bands ({bnumber}).")

            self.bname = bname if bname else [f"Band {i + 1}" for i in range(bnumber)]

        new_ds.FlushCache()

        # Optional: re-open the created raster as a Raster object
        return Raster(filename=output_filename)



    def extract_band(self, ofile=None, bnumber=None, bname=None, func=None, fformat=None):
        """
        Extracts specified bands from the current raster and saves them as separate raster files.

        Parameters
        ----------
        ofile : str
            Output file path for the extracted band.
        bnumber : list of int
            List of band indices (1-based) to extract.
        bname : list[str] | dict[int,str] | str | None
            Optional names for the bands. If str, it's used for all bands; supports templates
            like "MyBand_{num}" or "MyBand_{idx}".
        func : callable, optional
            Function to apply to the extracted band data.
        fformat : str, optional
            GDAL format for output.

        Returns
        -------
        list of str
            List of output file paths created.
        """
        if not bnumber:
            raise ValueError("bnumber (list of band indices) must be provided.")

        # --- normalize bname ---
        name_list = None
        name_map = None
        name_template = None

        if bname is None:
            pass
        elif isinstance(bname, (list, tuple)):
            name_list = list(map(str, bname))
        elif isinstance(bname, dict):
            # keys: band numbers (1-based), values: names
            name_map = {int(k): str(v) for k, v in bname.items()}
        elif isinstance(bname, str):
            name_template = bname  # may contain {num} or {idx}
        elif isinstance(bname, (int, float)):
            # Treat a single number as a single band name
            name_template = str(bname)
        else:
            raise TypeError("bname must be list/tuple, dict, str, or None.")

        output_files = []
        driver_format = fformat or self.fformat or "GTiff"

        for idx, num in enumerate(bnumber):
            if num < 1 or num > self.bnumber:
                raise IndexError(f"Band number {num} is out of range for this raster.")

            array = self.ds.GetRasterBand(num).ReadAsArray()

            # Resolve band name
            if name_list is not None:
                band_name_str = name_list[idx] if idx < len(name_list) else f"Band {num}"
            elif name_map is not None:
                band_name_str = name_map.get(num, f"Band {num}")
            elif name_template is not None:
                # allow {num} and/or {idx}; if no placeholders, reuse the same name or append suffix for multi
                if ("{num}" in name_template) or ("{idx}" in name_template):
                    band_name_str = name_template.format(num=num, idx=idx)
                else:
                    band_name_str = name_template if len(bnumber) == 1 else f"{name_template}_{num}"
            else:
                band_name_str = f"Band {num}"

            band_name = [band_name_str]  # assuming self.create expects a list

            # Resolve output path
            if ofile:
                output_path = Path(ofile)
                if len(bnumber) > 1:
                    file_path = output_path.with_name(f"{output_path.stem}_band{num}{output_path.suffix}")
                else:
                    file_path = output_path
            else:
                file_path = Path(f"extracted_band_{num}.tif")

            self.create(
                output_filename=str(file_path),
                array=array,
                xsize=self.xsize,
                ysize=self.ysize,
                bnumber=1,
                proj=self.proj.ExportToWkt() if self.proj else '',
                gtransform=self.gtransform,
                func=func,
                fformat=driver_format,
                bname=band_name
            )

            output_files.append(str(file_path))

        return output_files


    def translate_dtype(self, ofile, dtype=gdal.GDT_Float64, fformat=None):
        """
        Converts the raster to a new data type and saves it to a file.

        Parameters
        ----------
        ofile : str
            Path to output file.
        dtype : GDAL data type, optional
            New data type (default: gdal.GDT_Float64).
        fformat : str, optional
            GDAL format name (default: current format or 'GTiff').

        Returns
        -------
        Raster
            A new Raster object with the translated data type.
        """
        if self.ds is None:
            raise RuntimeError("No dataset loaded. Cannot translate data type.")

        driver_format = fformat or self.fformat or "GTiff"

        out_ds = gdal.Translate(
            ofile,
            self.ds,
            format=driver_format,
            outputType=dtype
        )

        if out_ds is None:
            raise RuntimeError("GDAL Translate failed.")

        return Raster(filename=ofile)


    # this functions only for ENVI formats
    #def get_envi_header(self, filename=None):
        #if filename:
        #    filename_stem = filename
        #else:
        #    filename_stem = self.path.as_posix()

        #header = HDRobject('.'.join([filename_stem, 'hdr']))
        #return header


    #def write_envi_header(self, filename, bn=None):

   #    with HDRobject('.'.join([filename, 'hdr'])) as hdr:
    #        if bn:
    #           if isinstance(bn, str):
    #                blist = [bn]
    #            elif isinstance(bn, list):
    #                blist = bn

    #            new_bandnames = [self.bname[i] for i in blist]
   #             print('Selected Bands', new_bandnames)
    #            hdr.band_names = new_bandnames
    #        else:
    #            hdr.band_names = self.bname
    #    hdr.write()
    #    hdr = None
        #filename = None
        #gdal.Unlink(filename)


    def close(self):
        del self.ds


    # was not used in this module
    def stack2vrt(self, band_list, ofile, xres=None, yres=None, nan=None, mask=None, overwrite=True):
        '''
        method to create a virtual linking to raster datasets.
        based on the source: https://gis.stackexchange.com/questions/44003/python-equivalent-of-gdalbuildvrt/314580
        VRT – GDAL Virtual Format


        Parameters:
        ----------
        band_list: list
                    list of the paths of individual bands
        ofile:  str
                    destination file name (full path)
        xres: str
                    see https://gdal.org/programs/gdalbuildvrt.html
                        https://gdal.org/drivers/raster/vrt.html
        yres: str

        nan: str

        overwrite: boolean
                    defines if the image should be overwritten

        Returns:
        -------

        Interesting links:
                    https://stackoverflow.com/questions/48706402/python-perform-gdalwarp-in-memory-with-gdal-bindings
        '''

        if xres and yres and nan:
            x_res = None
            y_res = None
            nan_value = np.NaN
            vrtres_code = None
            TargetAlignedPixels = False
        else:
            x_res = abs(self.xres)
            y_res = abs(self.yres)
            nan_value = self.nan
            vrtres_code = 'user'
            TargetAlignedPixels = True


        vrt = None
        # Link to modify the VRT file http://nagyak.eastron.hu/doc/gdal-1.6.2/docs/docs-64/gdal__vrttut.html
        # NOTE!! if bands of S1 rasters are in the same datetype, then they can be stacked together, otherwise use option -separate
        # addAlpha=False,
        vrt_options = gdal.BuildVRTOptions(resampleAlg=gdal.GRA_NearestNeighbour,
                                           separate=True,
                                           resolution=vrtres_code,
                                           xRes=x_res,
                                           yRes=y_res,
                                           targetAlignedPixels=TargetAlignedPixels,
                                           allowProjectionDifference=False,
                                           #outputBounds=bbox,
                                           VRTNodata=nan_value,
                                           srcNodata=nan_value)

        dst_file = Path(ofile)
        # check if the output file already exist => overwrite it
        if mask:
            BAND_LIST = band_list + [mask]
        else:
            BAND_LIST = band_list

        print('BAND_LIST', BAND_LIST)

        if dst_file.exists():
            if overwrite == True:
                vrt = gdal.BuildVRT(dst_file.as_posix(), BAND_LIST, options=vrt_options)
            else:
                raise ValueError('file already exist!')
        if not dst_file.exists() and (overwrite == True or overwrite == False):
            vrt = gdal.BuildVRT(dst_file.as_posix(), BAND_LIST, options=vrt_options)
        stack = vrt

        if mask:
            # TODO: check if mask has the same crs !
            # ------ 4. add mask Band as internal mask in VRT ------
            BandList = np.arange(1, vrt.RasterCount + 1, 1).tolist()
            print('BandList', BandList)

            options = gdal.TranslateOptions(maskBand=vrt.RasterCount,
                                            bandList=BandList,
                                            format='VRT',
                                           # projWin=[Raster(mask).minx, Raster(mask).maxy, Raster(mask).maxx, Raster(mask).miny],
                                           # projWinSRS=Raster(mask).proj,
                                            noData='none')  # bandList = [], stats = True,
            print(dst_file.parent)
            vrt_masked = gdal.Translate(destName=dst_file.parent.joinpath(dst_file.stem + '_masked.vrt').as_posix(), srcDS=vrt, options=options)
            vrt_masked.SetMetadataItem('dates',
                               '{}'.format(*BAND_LIST),
                               'TIMESERIES')
            stack = vrt_masked
        return stack  #object of <osgeo.gdal.Dataset>





    def reproject(self,
                  ofile,
                  nan=None,
                  to_crs=None,
                  bbox=None,
                  fformat=None,
                  resample_alg=gdal.GRA_NearestNeighbour,
                  reproj2image=None,
                  align2image=None,
                  cutlineDSName=None,
                  sql=None):
        """
        Reproject the raster to a new CRS or match a reference image grid.

        Parameters
        ----------
        ofile : str or Path
            Output file path.
        nan : float, optional
            NoData value.
        to_crs : int or str, optional
            Target EPSG code (e.g., 4326).
        bbox : list, optional
            [minx, miny, maxx, maxy] for output bounds.
        fformat : str, optional
            GDAL format (default: same as source or 'GTiff').
        resample_alg : GDAL resampling method
            e.g., gdal.GRA_Bilinear, gdal.GRA_NearestNeighbour.
        reproj2image : str, optional
            Path to a reference image to copy CRS/resolution from.
        align2image : str, optional
            Path to a reference image for pixel alignment.
        cutlineDSName : str, optional
            Path to vector cutline (shapefile, GeoJSON, etc.).
        sql : str, optional
            SQL query for cutline layer.

        Returns
        -------
        Raster
            New Raster object representing the reprojected file.
        """
        from pathlib import Path

        output_file = Path(ofile)
        targetAlignedPixels = False

        # Determine output CRS and resolution
        epsg = None
        x_res = y_res = None
        b_box = bbox
        driver_format = fformat or self.fformat or "GTiff"

        if to_crs:
            epsg = str(to_crs)
        elif reproj2image:
            ref_raster = Raster(reproj2image)
            epsg = str(ref_raster.epsg)
            x_res = ref_raster.xres
            y_res = ref_raster.yres
            b_box = b_box or ref_raster.bbox
            driver_format = driver_format or ref_raster.fformat
        elif align2image:
            align_raster = Raster(align2image)
            targetAlignedPixels = True
            epsg = str(align_raster.epsg)
            b_box = b_box or align_raster.bbox
            x_res = align_raster.xres
            y_res = align_raster.yres
            driver_format = driver_format or align_raster.fformat

        if not epsg:
            raise ValueError("Target CRS is not defined. Set 'to_crs', 'reproj2image', or 'align2image'.")

        # Determine NoData value
        nan_value = nan if nan is not None else self.nan or np.nan

        # Set GDAL Warp options
        options = gdal.WarpOptions(
            format=driver_format,
            cutlineDSName=cutlineDSName,
            cutlineSQL=sql,
            cropToCutline=True if cutlineDSName else False,
            outputBounds=b_box,
            xRes=x_res,
            yRes=y_res,
            dstSRS=f"EPSG:{epsg}",
            targetAlignedPixels=targetAlignedPixels,
            resampleAlg=resample_alg,
            dstNodata=nan_value
        )

        # Run reprojection
        result = gdal.Warp(destNameOrDestDS=output_file.as_posix(),
                           srcDSOrSrcDSTab=self.path.as_posix(),
                           options=options)

        if result is None:
            raise RuntimeError("gdal.Warp failed.")

        return Raster(filename=output_file)



    def resample_image(self,
                       ofile,
                       ref_image=None,
                       to_epsg='32633',
                       driver_format='GTiff',
                       cutlineDSName=None,
                       sql=None,
                       crop2cutline=True,
                       xres=None,
                       yres=None,
                       bbox=None,
                       resample_alg=gdal.GRA_Bilinear,
                       nan_value=None):
        """
        Resample and/or reproject the raster to match a reference grid or new resolution/CRS.

        Parameters
        ----------
        ofile : str
            Output file path.
        ref_image : str, optional
            Path to a reference raster to align to.
        to_epsg : str or int, optional
            Target EPSG code (default: '32633').
        driver_format : str, optional
            GDAL format (default: 'GTiff').
        cutlineDSName : str, optional
            Vector file for cropping (shapefile, GeoJSON) or .db - the sql for geometry extraction should be provided
        sql : str, optional
            SQL query for selecting cutline geometry.
        crop2cutline : bool, optional
            Whether to crop to cutline.
        xres, yres : float, optional
            Target pixel resolutions.
        bbox : list, optional
            Target bounding box.
        resample_alg : GDAL resampling method
            Interpolation method (default: gdal.GRA_Bilinear).
        nan_value : float, optional
            NoData value.

        Returns
        -------
        Raster
            Reprojected and/or resampled Raster object.
        """
        from pathlib import Path

        targetAlignedPixels = False
        epsg = str(to_epsg) if to_epsg else None

        if nan_value is None:
            nan_value = self.nan or np.nan

        if ref_image:
            ref_raster = Raster(filename=ref_image)
            targetAlignedPixels = True
            xres = xres or ref_raster.xres
            yres = yres or ref_raster.yres
            bbox = bbox or ref_raster.bbox
            epsg = epsg or ref_raster.epsg
            nan_value = nan_value or ref_raster.nan

            if str(ref_raster.epsg) != str(epsg):
                raise ValueError(f"EPSG mismatch: {ref_raster.epsg} != {epsg}")
        else:
            xres = xres or self.xres
            yres = yres or self.yres
            bbox = bbox or self.bbox
            if self.epsg and str(self.epsg) != str(epsg):
                raise ValueError(f"EPSG mismatch: {self.epsg} != {epsg}")

        if sql is not None:
            outputBounds = None  # Let GDAL crop via cutline
        else:
            outputBounds = bbox

        print(f"Resampling to EPSG:{epsg}, resolution: ({xres}, {yres})")

        options = gdal.WarpOptions(
            format=driver_format,
            cutlineDSName=cutlineDSName,
            cutlineSQL=sql,
            cropToCutline=crop2cutline,
            outputBounds=outputBounds,
            xRes=xres,
            yRes=yres,
            dstSRS=f"EPSG:{epsg}",
            targetAlignedPixels=targetAlignedPixels,
            resampleAlg=resample_alg,
            dstNodata=nan_value
        )

        ofile = Path(ofile)
        res_ds = gdal.Warp(ofile.as_posix(), self.path.as_posix(), options=options)

        if res_ds is None:
            raise RuntimeError("gdal.Warp failed during resample.")

        return Raster(filename=ofile)


    def RGB2LAB(self, band_number=[3, 2, 1]):

        values = {}
        # read selected band numbers
        Red, Green, Blue = self.ds.GetRasterBand(band_number[0]).ReadAsArray(), \
                  self.ds.GetRasterBand(band_number[1]).ReadAsArray(), \
                  self.ds.GetRasterBand(band_number[2]).ReadAsArray()

        values['RGB'] = cv2.merge([Red, Green, Blue])
        lab_image = color.rgb2lab(values['RGB'])
        #does not work for returns error
        #lab_image = cv2.cvtColor(values['RGB'], cv2.COLOR_RGB2LAB)

        lab_image_float = lab_image.astype(np.float32)
        values['L_lab'], values['A'], values['B'] = cv2.split(lab_image_float)

        #values['L'] = values['L'] / 255
        #values['A'] = values['A'] / 255
        #values['B'] = values['B'] / 255
        return values


    def RGB2LAB_block(self, block_R, block_G, block_B):
        """
        Convert a block of RGB data to LAB color space.

        Parameters:
            block_R, block_G, block_B (numpy.ndarray): Red, Green, Blue channel blocks.

        Returns:
            dict: A dictionary with 'L', 'A', and 'B' keys, each containing the corresponding LAB component block.
        """

        # Stack R, G, B into an RGB image and normalize to [0, 1] for skimage's rgb2lab function
        rgb_block = np.stack([block_R, block_G, block_B], axis=-1) / 255.0

        # Convert RGB block to LAB using skimage
        lab_block = color.rgb2lab(rgb_block)

        # Split LAB channels
        L, A, B = lab_block[:, :, 0], lab_block[:, :, 1], lab_block[:, :, 2]

        # Return the LAB components as a dictionary
        return {'L': L, 'A': A, 'B': B}


    def RGB2LUV(self, band_number=[3, 2, 1]):

        values = {}
        # read selected band numbers
        Red, Green, Blue = self.ds.GetRasterBand(band_number[0]).ReadAsArray(), \
                  self.ds.GetRasterBand(band_number[1]).ReadAsArray(), \
                  self.ds.GetRasterBand(band_number[2]).ReadAsArray()

        values['RGB'] = cv2.merge([Red, Green, Blue])
        #values['lab'] = color.rgb2lab(values['RGB'])
        luv_image = cv2.cvtColor(values['RGB'], cv2.COLOR_RGB2LUV)
        values['L'], values['U'], values['V'] = cv2.split(luv_image)

        return values



    def RGB2LUV_block(self, block_R, block_G, block_B):
        """
        Convert a block of RGB data to LUV color space and replace zeros with NaN.

        Parameters:
            block_R, block_G, block_B (numpy.ndarray): Red, Green, Blue channel blocks.

        Returns:
            dict: A dictionary with 'L', 'U', and 'V' keys, each containing the corresponding LUV component block.
        """

        # Mask where original inputs had NaN
        mask = np.isnan(block_R) | np.isnan(block_G) | np.isnan(block_B)

        # Fill NaNs with 0 temporarily for OpenCV (it can't handle NaN)
        R = np.nan_to_num(block_R, nan=0.0).astype(np.float32) / 255.0
        G = np.nan_to_num(block_G, nan=0.0).astype(np.float32) / 255.0
        B = np.nan_to_num(block_B, nan=0.0).astype(np.float32) / 255.0

        # Merge channels into RGB image
        rgb_block = cv2.merge([R, G, B])  # float32 in [0,1]

        # Convert to LUV
        luv_block = cv2.cvtColor(rgb_block, cv2.COLOR_RGB2LUV)

        # Split LUV channels
        L, U, V = cv2.split(luv_block)

        # Restore NaNs in the same positions as original input
        L[mask] = np.nan
        U[mask] = np.nan
        V[mask] = np.nan

        return {'L': L, 'U': U, 'V': V}



    @staticmethod
    def hsi(row, col, R, G, B):
        # Hue
        den = np.sqrt((R[row] - G[row]) ** 2 + (R[row] - B[row]) * (G[row] - B[row]))

        # Use ‘acosd’ function to find inverse cosine and obtain the result in degrees.
        # To avoid divide by zero exception add a small number in the denominator
        thetha = np.arccos(0.5 * (R[row] - B[row] + R[row] - G[row]) / den)  # Calculate the angle

        h = np.zeros(col)  # define temporary array

        # den>0 and g>= b element h is assigned thetha
        h[B[row] <= G[row]] = thetha[B[row] <= G[row]]
        # den>0 and the element h of g<= b is assigned thetha
        h[G[row] < B[row]] = 2 * np.pi - thetha[G[row] < B[row]]
        # den<0's element h is assigned a value of 0
        h[den == 0] = 0
        return h / (2 * np.pi)


    def RGB2HSI(self, band_number=[3, 2, 1]):
        '''

        based on Source https://www.imageeprocessing.com/2013/05/converting-rgb-image-to-hsi.html
        '''
        values = {}

        # read selected band numbers
        Red, Green, Blue = self.ds.GetRasterBand(band_number[0]).ReadAsArray(), \
                  self.ds.GetRasterBand(band_number[1]).ReadAsArray(), \
                  self.ds.GetRasterBand(band_number[2]).ReadAsArray()


        # Save the number of rows and columns of the original image
        row = np.shape(Red)[0]
        col = np.shape(Red)[1]

        # Each RGB component will be in the range of [0 255].  Represent the image in [0 1] range by dividing the image by 255.
        # represent the RGB image in [0 1] range
        R = np.where(Red != self.nan, Red / 255, self.nan)
        G = np.where(Green != self.nan, Green / 255, self.nan)
        B = np.where(Blue != self.nan, Blue / 255, self.nan)

        # define h channel
        H = np.zeros((row, col))
        # define s channel
        S = np.zeros((row, col))

        for i in range(row):
            H[i] = Raster.hsi(i, col, R, G, B) # Assign to h channel after radian

        # Divide the hue component by 360 to represent in the range [0 1]
        # Normalize to the range [0 1]
        values['H'] = H * 255


        # Saturation
        for i in range(row):
            min = []
            # Find the minimum value of each group of rgb values
            for j in range(col):
                arr = [B[i][j], G[i][j], R[i][j]]
                min.append(np.min(arr))
            min = np.array(min)
            S[i] = 1 - min * 3 / (R[i] + G[i] + B[i])
            S[i][R[i] + B[i] + G[i] == 0] = 0


        values['S'] = S * 255

        # Intensity
        values['I'] = ( (R+G+B) / 3 ) * 255

       # values['HS'] = cv2.merge([values['H'], values['S']])

        # HSI
        return values



    def RGB2HSI_block(self, block_R, block_G, block_B):
        """
        Convert a block of RGB data to HSI color space.

        Parameters:
            block_R, block_G, block_B (numpy.ndarray): Red, Green, Blue channel blocks.

        Returns:
            dict: A dictionary with 'H', 'S', and 'I' keys, each containing the corresponding HSI component block.
        """
        # Block dimensions
        row, col = block_R.shape

        # Normalize RGB channels to [0, 1] range
        R = np.where(block_R != self.nan, block_R / 255, self.nan)
        G = np.where(block_G != self.nan, block_G / 255, self.nan)
        B = np.where(block_B != self.nan, block_B / 255, self.nan)

        # Initialize H, S, I channels
        H = np.zeros((row, col))
        S = np.zeros((row, col))
        I = np.zeros((row, col))

        # Hue calculation
        for i in range(row):
            for j in range(col):
                r, g, b = R[i, j], G[i, j], B[i, j]
                num = 0.5 * ((r - g) + (r - b))
                denom = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6  # Add small value to avoid division by zero
                theta = np.arccos(num / denom) if denom != 0 else 0

                # Determine hue based on RGB values
                if b <= g:
                    H[i, j] = theta
                else:
                    H[i, j] = 2 * np.pi - theta

        # Normalize H to [0, 255]
        H = (H / (2 * np.pi)) * 255

        # Saturation calculation
        for i in range(row):
            for j in range(col):
                min_val = min(R[i, j], G[i, j], B[i, j])
                S[i, j] = 1 - (3 * min_val / (R[i, j] + G[i, j] + B[i, j] + 1e-6))  # Avoid division by zero
                if R[i, j] + G[i, j] + B[i, j] == 0:
                    S[i, j] = 0  # Set saturation to 0 if R + G + B is zero

        # Scale S to [0, 255]
        S *= 255

        # Intensity calculation
        I = (R + G + B) / 3 * 255  # Scale I to [0, 255]

        # Return the HSI components as a dictionary
        return {'H': H, 'S': S, 'I': I}


    def calculate_lbp(self, p, r, band_number=[3, 2, 1]):
        values = {}
        from skimage.feature import local_binary_pattern
        from skimage.feature import local_binary_pattern

        # read selected band numbers
        R, G, B = self.ds.GetRasterBand(band_number[0]).ReadAsArray(), \
                  self.ds.GetRasterBand(band_number[1]).ReadAsArray(), \
                  self.ds.GetRasterBand(band_number[2]).ReadAsArray()

        values['BGR'] = cv2.merge([B, G, R])

        # Convert the RGB image to grayscale
        gray_image = cv2.cvtColor(values['BGR'], cv2.COLOR_BGR2GRAY)
        # Calculate LBP
        lbp = local_binary_pattern(gray_image, P=p, R=r, method='uniform')

        return lbp


    # use formula
    def RGB2YCbCr(self, band_number=[3, 2, 1]):
        values = {}
        R = self.ds.GetRasterBand(band_number[0]).ReadAsArray().astype(float)
        G = self.ds.GetRasterBand(band_number[1]).ReadAsArray().astype(float)
        B = self.ds.GetRasterBand(band_number[2]).ReadAsArray().astype(float)

        nan = self.nan

        mask = (R != nan) & (G != nan) & (B != nan)

        Y = np.full(R.shape, nan, dtype=float)
        Cb = np.full(R.shape, nan, dtype=float)
        Cr = np.full(R.shape, nan, dtype=float)

        # Apply formulas only where valid
        Y[mask] = 0.299 * R[mask] + 0.587 * G[mask] + 0.114 * B[mask]
        Cb[mask] = -0.168736 * R[mask] - 0.331264 * G[mask] + 0.5 * B[mask] + 128
        Cr[mask] = 0.5 * R[mask] - 0.418688 * G[mask] - 0.081312 * B[mask] + 128

        values['Y'] = Y
        values['Cb'] = Cb
        values['Cr'] = Cr

        return values


    def RGB2YCbCr_block(self, R, G, B):
        '''
        uses ITU-R BT.601 standard formula. Operates on floating-point data, assuming RGB values are in [0, 255].

        :param R:
        :param G:
        :param B:
        :return:
        '''
        R = R.astype(float)
        G = G.astype(float)
        B = B.astype(float)

        Y = np.full(R.shape, np.nan, dtype=float)
        Cb = np.full(R.shape, np.nan, dtype=float)
        Cr = np.full(R.shape, np.nan, dtype=float)

        # Valid data mask
        mask = ~(np.isnan(R) | np.isnan(G) | np.isnan(B))

        # Apply transformation only on valid pixels
        Y[mask] = 0.299 * R[mask] + 0.587 * G[mask] + 0.114 * B[mask]
        Cb[mask] = -0.168736 * R[mask] - 0.331264 * G[mask] + 0.5 * B[mask] + 128
        Cr[mask] = 0.5 * R[mask] - 0.418688 * G[mask] - 0.081312 * B[mask] + 128

        return {'Y': Y, 'Cb': Cb, 'Cr': Cr}


    def RGB2YCbCr_block_(self, R, G, B):
        '''
        uses OpenCV's built-in,  operates on 8-bit integers

        :param R:
        :param G:
        :param B:
        :return:
        '''
        # Stack into RGB image (OpenCV uses uint8 in [0,255] or float32 in [0,1])
        rgb = cv2.merge([R, G, B]).astype(np.uint8)

        # Convert using OpenCV
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)  # shape: (rows, cols, 3)

        # Split channels
        Y, Cr, Cb = cv2.split(ycrcb)

        return {'Y': Y.astype(np.float32), 'Cb': Cb.astype(np.float32), 'Cr': Cr.astype(np.float32)}



    # Does not WORK proper!!!!
    @staticmethod
    def reclassify(array, thresholds):
        print(array)
        print(type(array))
        if len(thresholds) == 1:
            i = 0
            thresh = thresholds[i]
            print(i, thresh)
            array[(array <= thresh)] = i+1
            array[(array > thresh)] = i+2
            return array
        if len(thresholds) > 1:
            i = 0
            while i < len(thresholds):
                thresh = thresholds[i]
                print('inside', i, array, i+1)
                array[(array < thresh)] = i+1
                i = i + 1
            i = len(thresholds)
            thresh = thresholds[i - 1]
            print(i, thresh)
            array[(array > thresh)] = i + 1
            print('outside', i, array, i + 1)
            return array


    def write(self,
              new_ds,
              array=None,
              bnumber=None,
              nan=None
              ):

        b_number = bnumber or self.bnumber
        nan_value = nan if nan is not None else self.nan

        if isinstance(array, np.ndarray):
            if array.ndim == 3:
                # Array shape: (bands, rows, cols) or (rows, cols, bands)
                for b in range(1, b_number + 1):
                    band_array = array[b - 1]
                    new_ds.GetRasterBand(b).WriteArray(band_array)
            elif array.ndim == 2:
                new_ds.GetRasterBand(1).WriteArray(array)
            else:
                raise ValueError("Unsupported ndarray shape.")

        elif isinstance(array, dict):
            for b in range(1, b_number + 1):
                band_array = array.get(b)
                if band_array is None:
                    raise ValueError(f"Missing band {b} in array dict.")
                new_ds.GetRasterBand(b).WriteArray(band_array)
        else:
            raise TypeError("Array must be a 2D/3D ndarray or a dict of arrays.")

        # Set NoData
        for b in range(1, b_number + 1):
            new_ds.GetRasterBand(b).SetNoDataValue(nan_value)

        new_ds.FlushCache()

        return new_ds


    def write_block(self, new_ds, array, x_offset, y_offset, bnumber=1, nan=None):
        """
        Write a block (chunk) of data to a specific location within the raster dataset.

        Parameters:
            new_ds: The GDAL dataset object for writing.
            array (np.ndarray): The data array to write (2D block).
            x_offset (int): X offset (column) where the block should be written.
            y_offset (int): Y offset (row) where the block should be written.
            bnumber (int): Band number to write to (default is 1).
            nan (float or None): Value to treat as NaN, if applicable.
            driver_format (str): The format for the GDAL driver (default is 'GTiff').
        """
        if nan is not None:
            # Replace NaN values in array with a specific value if provided
            array = np.where(np.isnan(array), nan, array)

        # Write the array block to the specified position in the raster
        band = new_ds.GetRasterBand(bnumber)
        band.WriteArray(array, xoff=x_offset, yoff=y_offset)
        band.FlushCache()


    def create_ds(self,
              output_filename,
              array=None,
              xsize=None,
              ysize=None,
              bnumber=None,
              bname=None,
              proj=None,
              gtransform=None,
              func=None,
              fformat=None,
              nan=None,
              datatype=gdal.GDT_Float64
              ):
        '''
        function creates new raster based on entered parameters
        calling this function will write the result to a file

        Parameters
        ----------
        output_filename: str
        xsize:
        ysize:
        bnumber:
        proj:
        gtransform:
        func: function
            function to manipulate the array values. examp: func=convert2db()
        fformat:

        Returns
        -------

        '''

        if fformat:
            driver_format = fformat
        else:
            driver_format = self.fformat

        if xsize and ysize:
            x_size = xsize
            y_size = ysize
        else:
            x_size = self.xsize
            y_size = self.ysize
        if bnumber:
            b_number = bnumber
        else:
            b_number = self.bnumber
        if proj:
            crs_proj = proj
        else:
            crs_proj = self.ds.GetProjection()
        if gtransform:
            g_transform = gtransform
        else:
            g_transform = self.gtransform
            # TODO: define self.datatime =  !!!!!
        data_type = datatype

        driver = gdal.GetDriverByName(driver_format)
        new_ds = driver.Create(output_filename, x_size, y_size, b_number, data_type)  # gdal.GDT_Byte
        new_ds.SetProjection(crs_proj)
        new_ds.SetGeoTransform(g_transform)

        return new_ds

    # this is used to create an artifical sample patch for CNN-Analysis in eCognition
    def create_ds_no_crs(self,
                         output_filename,
                         xsize=None,
                         ysize=None,
                         bnumber=None,
                         fformat=None,
                         datatype=gdal.GDT_Float64,
                         x_tiles_number=None,
                         y_tiles_number=None,
                         proj=None,
                         gtransform=None):
        """
        Create an artificial raster for CNN patches, CRS is set but should be ignored as it is invalid.
        The raster is filled with NoData and aligned synthetically.

        Parameters
        ----------
        output_filename : str
            Output raster path.
        xsize, ysize : int
            Base size of each tile.
        bnumber : int
            Number of bands.
        fformat : str
            GDAL format.
        datatype : GDAL dtype
            Data type of the raster.
        x_tiles_number, y_tiles_number : int
            Number of tiles in each direction.
        proj : str, optional
            Projection WKT (if any).
        gtransform : list, optional
            Custom GeoTransform.

        Returns
        -------
        GDAL Dataset
            The created (empty) raster.
        """

        driver_format = fformat or self.fformat or "GTiff"
        driver = gdal.GetDriverByName(driver_format)

        xsize = xsize or self.xsize
        ysize = ysize or self.ysize
        bnumber = bnumber or self.bnumber
        x_tiles_number = x_tiles_number or 1
        y_tiles_number = y_tiles_number or 1
        data_type = datatype
        nodata = np.nan

        total_x = xsize * x_tiles_number
        total_y = ysize * y_tiles_number + (2 * xsize)

        print(f"Creating raster: {output_filename}")
        print(f"Size: {total_x} x {total_y}, Bands: {bnumber}, Type: {data_type}")
        print(f"Driver: {driver.ShortName if driver else 'None'}")

        new_ds = driver.Create(output_filename, total_x, total_y, bnumber, data_type)

        for i in range(1, bnumber + 1):
            new_ds.GetRasterBand(i).SetNoDataValue(nodata)

        if gtransform:
            geo_transform = gtransform
        else:
            ulx, uly = 0, total_y
            geo_transform = [ulx, 1, 0, uly, 0, -1]

        new_ds.SetGeoTransform(geo_transform)

        if proj:
            new_ds.SetProjection(proj)

        return new_ds


    def enhancement(self, bnumber=[3, 2, 1], channel='Y', equalize_method='adaptive'):
        '''
        additional function to apply image enhancement

        first converts the RGB to YCbCr, then apply enhancement
        '''

        image = Raster.RGB2YCbCr(self, band_number=bnumber)

        # Histogram matching
        # image_array = exposure.match_histograms(YY, Cr)

        if equalize_method =='global':
            # create mask for nan values
            mask = np.isfinite(image[channel])
            image_array = exposure.equalize_hist(image[channel], nbins=256, mask=mask)

        if equalize_method == 'adaptive':
            # Adaptive Equalization
            image_array = exposure.equalize_adapthist(image[channel], clip_limit=0.03, nbins=256)
        elif equalize_method == 'root':
            image_array = np.sqrt(image[channel])
        elif equalize_method == 'log':
            image_array = np.log(image[channel])
        elif equalize_method == 'None':
            image_array=image[channel]

        # create a CLAHE object (Arguments are optional).
        # lab_planes = cv2.split(YY*255)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        # lab_planes[0] = clahe.apply(lab_planes[0])
        # image_array = cv2.merge(lab_planes)

        # create mask for nan values
        mask = np.isfinite(image[channel])

        # apply mask of nan values to mask invalid values after equalization
        # img = np.where(mask, np.log(np.sqrt(image_array)) / 255, np.nan)
        img = np.where(mask,  image_array / 255, np.nan)
        return img



def combine_channels_to_multiband(output_filename, channels, dtype=gdal.GDT_Float32):
    """
    Combine individual channel files into a single multi-band file.

    Parameters:
        output_filename (str): The filename for the combined output.
        channels (list of str): List of channel filenames in the order they should appear in the output.
        dtype: The GDAL data type for each band (default is Float32).
    """
    # Open the first channel to get dimensions
    src_ds = gdal.Open(channels[0])
    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    projection = src_ds.GetProjection()
    geotransform = src_ds.GetGeoTransform()

    # Create the output multi-band file
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_filename, cols, rows, len(channels), dtype)
    out_ds.SetProjection(projection)
    out_ds.SetGeoTransform(geotransform)

    # Write each channel to a separate band
    for idx, channel_file in enumerate(channels):
        print(idx, channel_file)
        channel_ds = gdal.Open(channel_file)
        out_ds.GetRasterBand(idx + 1).WriteArray(channel_ds.ReadAsArray())
        channel_ds = None  # Close individual channel file

    out_ds = None  # Close combined output file
    print(f"Combined channels into {output_filename}")