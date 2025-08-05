"""
3. Script to join together the sample patches croped with script crop_uav_samples.py to one raster file with xml metadata file

contains additional functions
collect_raster_samples()
add_empty_2Darrays()
stack_vertical_samples()
"""

from pathlib import Path
from osgeo import gdal as gdal
from geoutils.raster import Raster
import numpy as np
import dask.array as da
import config as cfg
import sys
from create_CNN_samples.xml_utils import generate_xml_meta



import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def collect_raster_samples(raster_filenames, raster_dir, band_map):
    '''
    Collects 2D (30x30) raster samples and aggregates them per band into a dictionary.

    Parameters:
        raster_filenames (list): List of raster file names (relative to `raster_dir`).
        raster_dir (str or Path): Base directory containing the raster files.
        band_map (dict): Keys are band indices (1-based), values can be names or identifiers.

    Returns:
        dict: Keys are band indices, values are lists of Dask arrays (one per valid sample).
    '''

    sample_data = {band_idx: [] for band_idx in band_map.keys()}

    for filename in raster_filenames:
        raster_path = Path(raster_dir) / filename
        raster_file = Raster(raster_path.as_posix())

        logger.debug(f"Processing file: {raster_path.name}")
        logger.debug(f"Size: {raster_file.xsize}, Resolution: {raster_file.xres}, Bands: {raster_file.bnumber}")

        if raster_file.bnumber != len(band_map):
            logger.error("Mismatch between expected and actual number of bands!")
            sys.exit("Invalid band count in raster.")

        sample_array = raster_file.ds.GetRasterBand(1).ReadAsArray()

        if np.isnan(sample_array).any():
            logger.warning(f"Skipping {filename}: contains NaN values.")
            continue

        if sample_array.shape != (30, 30):
            logger.warning(f"Skipping {filename}: invalid shape {sample_array.shape}.")
            continue

        for band_idx in band_map.keys():
            band_array = raster_file.ds.GetRasterBand(band_idx).ReadAsArray()
            dask_band = da.from_array(band_array, chunks='auto')
            sample_data[band_idx].append(dask_band)
            logger.debug(f"Added band {band_idx} from {filename}.")

        raster_file.close()

    return sample_data

def add_empty_2Darrays(sample_data, empty_array, target_sample_count):
    '''
    Complements raster sample strips with empty (30x30) arrays to ensure uniform length for stacking.

    Parameters:
        sample_data (dict): Dictionary of band -> list of 2D Dask arrays (from collect_raster_samples).
        empty_array (np.ndarray): A (30, 30) NumPy array representing a blank sample.
        target_sample_count (int): Desired number of samples per band.

    Returns:
        dict: Updated sample_data with empty arrays appended where needed.
    '''

    for band, samples in sample_data.items():
        current_count = len(samples)
        missing_count = target_sample_count - current_count

        if missing_count > 0:
            for _ in range(missing_count):
                samples.append(da.from_array(empty_array, chunks='auto'))

    return sample_data

def stack_vertical_samples(sample_data):
    '''
    Vertically stacks a list of 2D sample arrays per band.

    Parameters:
        sample_data (dict): Dictionary where each key is a band index or name,
                            and each value is a list of 2D Dask arrays.

    Returns:
        dict: Dictionary with the same keys, where each value is a single 2D Dask array
              (stacked vertically, shape = (30 * n_samples, 30)).
    '''

    stacked_data = {}

    for band_key, band_arrays in sample_data.items():
        if not band_arrays:
            logger.warning(f"No arrays found for band {band_key}; skipping.")
            continue

        try:
            stacked = da.vstack(band_arrays)
            stacked_data[band_key] = stacked.rechunk(chunks='auto')

            logger.debug(f"Stacked {len(band_arrays)} arrays for band {band_key}. Result shape: {stacked.shape}")
        except Exception as e:
            logger.error(f"Failed to stack arrays for band {band_key}: {e}")

    return stacked_data


pattern_decode = {
                  '2_0_1': 'vital_crop'
                , '3_0_1': '_stem_elongation'
                , '4_0_1': '_heading'
                , '5_0_1': 'flowering_crop'
                , '6_0_1': '_ripening'
                , '7_0_1': 'dry_crop'

                , '2_1_1': 'vital_lodged_crop'
                , '3_1_1': 'vital_lodged_crop'
                , '4_1_1': 'vital_lodged_crop'
                , '5_1_1': 'vital_lodged_crop'
                , '6_1_1': 'dry_lodged_crop'
                , '7_1_1': 'dry_lodged_crop'

                , '0_0_1': 'bare_soil'
                , '0_0_2': 'bare_soil'
                , '1_0_2': 'bare_soil'
                , '2_0_2': 'bare_soil'
                , '3_0_2': 'bare_soil'
                , '4_0_2': 'bare_soil'
                , '5_0_2': 'bare_soil'
                , '6_0_2': 'bare_soil'
                , '7_0_2': 'bare_soil'

                , '1_0_3': 'weed_infestation'
                , '2_0_3': 'weed_infestation'
                , '3_0_3': 'weed_infestation'
                , '4_0_3': 'weed_infestation'
                , '5_0_3': 'weed_infestation'
                , '6_0_3': 'weed_infestation'
                , '7_0_3': 'weed_infestation'

                , '7_1_3': 'weed_infestation' # weed infestation on lodged crop
                }

RGB3 = {
          1: 'B'
        , 2: 'G'
        , 3: 'R'
        }

CIR5 = {
          1: 'B'
        , 2: 'G'
        , 3: 'R'
        , 4: 'RE'
        , 5: 'NIR'
        }

CIR16 = {1: 'A',
       2: 'B',
       3: 'B1',
       4: 'B2',
       5: 'B3',
       6: 'B4',
       7: 'B5',
       8: 'Cb',
       9: 'Cr',
       10: 'H',
       11: 'I',
       12: 'L',
       13: 'S',
       14: 'U',
       15: 'V',
       16: 'Y'
       }

RGB14 = {1: 'A',
       2: 'B',
       3: 'B1',
       4: 'B2',
       5: 'B3',
       #6:'B4',
       #7:'B5',
       6: 'Cb',
       7: 'Cr',
       8: 'H',
       9: 'I',
       10: 'L',
       11: 'S',
       12: 'U',
       13: 'V',
       14: 'Y'
       }



# Select the Input Parameters
# RGB3  is the color infrared UAV Image that contains of 3 channels R, G, B (only DN)
# CIR5  is the color infrared UAV Image that contains of 5 channels R, G, B, NIR, RE (only reflectance)

# CIR16   is the color infrared UAV Image that contains of 5 channels R, G, B, NIR, RE (only reflectance) + color spaces derived from this channels
# RGB14   is the rgb UAV Image that contains of 3 channels R, G, B (DN and reflectance) + color spaces derived from this channels

CHANNELS = CIR5 # RGB3 # CIR16 # RGB14
sensor = 'CIR' # 'RGB'



#in_file = r'M:\__crop_samples\CIR_RP_16_samples_tiles_GeneralisedSAMPLES\_WW'
#output = Path(r'M:\__crop_samples')

in_file = Path(cfg.testfolder_path).joinpath('samples_tiles\CIR\MRKN')
output = Path(cfg.testfolder_path).joinpath('__crop_samples')
output.mkdir(parents=True, exist_ok=True)




#--------- Collect information on patterns (folder) and images (fields)
dict_of_images = {}
x_axis_length, y_axis_length = [], []

for i in in_file.iterdir():
    # x direction
    if i.is_dir():
        print('Found directory: {}'.format(i))
        x_axis_length.append(i)
        # y direction
        file_list = []
        for jdx, file in enumerate(i.glob('*.tif')):
            if jdx == 0:
                default_file = file
            fid, crop_type, bbch_group, damage, pattern, tile_idx = file.stem.split('_')
            file_list.append(file)
        #dict_of_images['{}_{}_{}_{}_{}'.format(fid, crop_type, bbch_group, damage, pattern)] = file_list
        dict_of_images['{}_{}_{}_{}_{}'.format(pattern, damage, bbch_group, crop_type, fid)] = file_list
        y_axis_length.append(jdx+1)



# store crop type for outputfile
CROP_TYPE1 = Path(in_file).stem
print(CROP_TYPE1)



#--------- Create an empty GDAL dataset to store the sample patch matrix

# define destination file
dst_file = output.joinpath('{}_{}_sample_composit.tif'.format(CROP_TYPE1, sensor))
ifile = Raster(default_file.as_posix())
BAND_NUMBER = ifile.bnumber

max_x_axis = len(x_axis_length)
max_y_axis = max(y_axis_length)


# overwrite sample patch matrix if exist
if dst_file.exists():
    print(dst_file)
    dst_file.unlink()

ds_transform = ifile.create_ds_no_crs(output_filename=dst_file.as_posix(),
                                                                  bnumber=BAND_NUMBER,
                                                                  fformat='GTiff',
                                                                  x_tiles_number=max_x_axis * 2,
                                                                  y_tiles_number=max_y_axis,
                                                                  datatype=gdal.GDT_Float32
                                                                  )


# create an empty array to even out the columns
empty_img_array = np.zeros((ifile.xsize, ifile.ysize), dtype=np.float32)#, dtype=np.uint32
empty_img_array[empty_img_array==0] = np.nan # use this to set to nan values



# initial coordinate upperleft
x = 0
y = ifile.ysize

data_list = {} # all samples will be collected here
for i in CHANNELS.keys():
    data_list[i] = []

band_storage = {}
meta_collect = []
pattern_before = None
pat_before, damage_before, bbch_before = None, None, None

# concatenate each vertical composit () together horizontal
for i, (pattern, f_list) in enumerate(sorted(dict_of_images.items())):
    #print(pattern)

    #fid, crop_type, bbch_group, damage, pat = pattern.split('_')
    pat, damage, bbch_group, crop_type, fid = pattern.split('_')
    #print(pat, damage, bbch_group)


    # collect samples vertically -> Level 2 sample definiton (e.g. 2_0_1 from different fields and study areas samples under the same label)
    v_sample_collection = collect_raster_samples(raster_filenames=f_list, raster_dir=in_file, band_map=CHANNELS)

    v_sample_strip = add_empty_2Darrays(sample_data=v_sample_collection, empty_array=empty_img_array, target_sample_count=max_y_axis)
    vertical_samples = stack_vertical_samples(v_sample_strip)

    for band, pattern_array in vertical_samples.items():
        print('p_array_shape: {}'.format(pattern_array.shape))

        # add the empty column to separate between patterns -> Level 1 sample definiton (e.g. vital_crop, bare_soil etc.. in total 7 labels)
        if ((pat != pat_before) or (damage != damage_before)) or ((pat != pat_before) and (damage != damage_before)):

            x = ['{}_{}_{}_{}_{}'.format(crop_type, bbch_group, damage, pat, fid),
                 ['empty column']]

            #collect samples horizontally -> Level 2 sample definiton (e.g. 2_0_1, 2_1_1 ... ) is concatenated horizontally with empty column between  Level 1 sample definiton (e.g. vital_crop, bare_soil etc.. in total 7 labels)
            emp_array = np.zeros((pattern_array.shape[0], pattern_array.shape[1]))
            emp_array[emp_array == 0] = np.nan
            empty_img_vcomposit_array = da.from_array(emp_array)
            p_array = da.hstack((empty_img_vcomposit_array, pattern_array))

            # collect for metainfo in xml file
            if x not in meta_collect:
                meta_collect.append(x)
        else:
            p_array = pattern_array

        for k, v in data_list.items():
            if band == k:
                v.append(p_array)

    meta_collect.append(
        ['{}_{}_{}_{}_{}'.format(crop_type, bbch_group, damage, pat, fid), [f.as_posix() for f in f_list]])

    # to add the empty column to separate the patterns
    pattern_before = pattern
    pat_before = pat
    damage_before = damage
    bbch_before = bbch_group


# collect metadata
meta_dict = {}
for i, v in enumerate(meta_collect):
    meta_dict[i] = v


# write meta information to a file
meta_file = output.joinpath('{}_{}_sample_composit.xml'.format(CROP_TYPE1, sensor))

# this is an old version of the meta file in TXT, new version in XML
"""
BANDS = np.arange(1, BAND_NUMBER + 1, 1).tolist()

for i in np.arange(1, BAND_NUMBER + 1, 1).tolist():
    print(CHANNELS[i])

with open(meta_file, 'w') as f:
    f.write('Meta-information for a raster file: ' + dst_file.as_posix() + '\n\n')
    if ifile.bnumber == len(CHANNELS.keys()):
        band_names = [CHANNELS[i] for i in np.arange(1, BAND_NUMBER + 1, 1).tolist()]
        f.write('Band number: {}'.format(CHANNELS.keys()) + '\n\n')
        f.write('Band name: {}'.format(band_names) + '\n\n')
    else:
        f.write('Band number: {}'.format(BANDS) + '\n\n')
        f.write('Band name:  {}'.format(BANDS) + '\n\n')
    f.write(
        'number of column (1 Chessboard Segment)' + ' >>> ' + ' sl_nr: ' + ' >>> ' + 'Pattern Code' + ' >>> '
        + 'Number of the Samples (1 Chessboard Segment) (rows)' + '\n\n')
    for k, v in meta_dict.items():
        f.write(str(k) + ' >>> ' + str('_'.join(v[0].split('_')[:-1])) + ' >>> ' + ' sl_nr: ' + str(
            '_'.join(v[0].split('_')[-1:])) + ' >>> ' + str(len(v[1])) + ' samples ' + '\n\n')
f.close()



meta_file = output.joinpath('{}_{}_sample_composit_full.txt'.format(CROP_TYPE1, sensor))
with open(meta_file, 'w') as f:
    f.write('Meta-information for a raster file: ' + dst_file.as_posix() + '\n\n')
    if ifile.bnumber == len(CHANNELS.keys()):
        band_names = [CHANNELS[i] for i in np.arange(1, BAND_NUMBER + 1, 1).tolist()]
        f.write('Band number: {}'.format(CHANNELS.keys()) + '\n\n')
        f.write('Band name: {}'.format(band_names) + '\n\n')
    else:
        f.write('Band number: {}'.format(BANDS) + '\n\n')
        f.write('Band name:  {}'.format(BANDS) + '\n\n')
    f.write(
        'number of column (1 Chessboard Segment)' + ' >>> ' + ' sl_nr: ' + ' >>> ' + 'Pattern Code' + ' >>> '
        + 'Number of the Samples (1 Chessboard Segment) (rows)' + '\n\n')
    f.write('   ' * 15 + 'number of row (1 Chessboard Segment)' + ' : ' + 'File Names of the Sample ' + '\n\n')
    for k, v in meta_dict.items():
        f.write(str(k) + ' >>> ' + str('_'.join(v[0].split('_')[:-1])) + ' >>> ' + ' sl_nr: ' + str(
            '_'.join(v[0].split('_')[-1:])) + ' >>> ' + str(len(v[1])) + ' samples ' + '\n\n')
        for idx, value in enumerate(v[1]):
            f.write('   ' * 15 + str(idx) + ' : ' + str(value) + '\n\n')
f.close()
"""


for k, band_array in data_list.items():
    print('p_array_shape2: {}'.format(band_array[0].shape))
    # band_array.compute_chunk_sizes()
    # array = da.concatenate(band_array, axis=1)

    array = da.hstack(band_array)
    print('p_array_shape3:{} '.format(array[1].shape))
    numpy_array = np.asarray(array.rechunk(chunks='auto'))
    band_storage[k] = numpy_array



# band_storage = {}
# if not Path(dst_file).exists():
# for k, band_array in enumerate([blue, green, red, redge, nired]):
#    band_storage[k] = np.concatenate(band_array[:], axis=0)
# print(np.concatenate(band_array[:]))
# print('--' * 10)


for k, v in band_storage.items():
    print(k, v.shape)
    print(v)


ifile.write(output_filename=dst_file.as_posix(),
            new_ds=ds_transform,
            array=band_storage,
            bnumber=BAND_NUMBER,
            # nan=None,
            driver_format='GTiff')





#---------  collect meta information from created sample patch matrix to a dictionary

segment_dict = {}
for column_position, v in meta_dict.items():
    segment_meta = {}

    crop_type_code, bbch_group, damage, pattern, fid = v[0].split('_')
    segment_paths = v[1]

    #print(pattern_code)
    #print(segment_path)
    #field_amount = len(segment_paths)

    segment_meta['crop_type_code'] = crop_type_code
    segment_meta['damage'] = damage # lodged / none lodged
    segment_meta['pattern_code'] = '{}_{}_{}_{}'.format( crop_type_code, bbch_group, damage, pattern)
    segment_meta['pattern_subcode'] = '{}_{}_{}'.format(bbch_group, damage, pattern)
    segment_meta['pattern'] = pattern_decode['{}_{}_{}'.format(bbch_group, damage, pattern)]
    segment_meta['flink'] = segment_paths


    fid_set = set()
    for i, path in enumerate(segment_paths):
        fid = Path(path).name.split('_')[0]
        fid_set.add(fid)


    segment_meta['fid_list'] = list(fid_set)
    segment_meta['fid_amount'] = len(fid_set)

    segment_dict[column_position] = segment_meta


    fid_paths = {}
    for fid in fid_set:
        for i, path in enumerate(segment_paths):
            collection = Path(path).parent.glob("{}*.tif".format(fid))
        fid_paths[fid] = [i.as_posix() for i in list(collection)]
    segment_meta['fid_path_collection'] = fid_paths



# read some metainfo from created file
ifile = Raster(dst_file.as_posix())

raster_meta = {}
raster_meta['size'] = (ifile.xsize, ifile.ysize)
raster_meta['dtype'] = ifile.dtype_as_str
raster_meta['fpath'] = ifile.path.as_posix()
raster_meta['bnumber'] = str(ifile.bnumber)


# create metadata file for composit of sample patches
generate_xml_meta(
                    raster=raster_meta,
                    chanell_meta=CHANNELS,
                    meta_dict=segment_dict,
                    ofile=meta_file
                    )