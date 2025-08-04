"""
script to read  raster file and convert to datatype 32float if raster is datatype 64 float
this was required for some UAV images that have been exported as float64

"""
from pathlib import Path
from osgeo import gdal
print(gdal.__version__)
from geoutils.raster import Raster
import config as cfg


#https://gist.github.com/CMCDragonkai/ac6289fa84bcc8888035744d7e00e2e6
decoding_of_gdal_formats = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}


root = cfg.testfolder_path
#tails = ['A', 'B', 'V', 'U', 'L', 'I', 'S', 'H', 'Cr', 'Cb', 'Y']

in_file = Path(root)
files = in_file.glob("**/*.tif")

for i in in_file.iterdir():
    if i.is_dir():
        for jdx, file in enumerate(i.iterdir()):
            if file.is_file():

                print(file)
                raster = Raster(file.as_posix())

                # convert to float 32 (7) if dtype is float 64 (6)
                if raster.dtype == 7: # 7 -> 64 float

                    output_filename = file.parent.joinpath('32_bit')
                    if not output_filename.exists():
                        output_filename.mkdir(parents=True, exist_ok=True)

                    raster.translate_dtype(output_filename.joinpath(file.stem + '_32bit.tif').as_posix(), dtype=gdal.GDT_Float32)
                    raster.close()

