"""
script to calculate semivariogram for high resolution UAV Images
to evaluate texture/contextual information of the sample patches

it also have functions to calculate Local Binary Pattern (LBP) amd gray-level co-occurrence matrix (GLCM) and
plot semivariogram on top of that
The semivariogram RMSE parameters can be entered into DB to select the best fit and semivariogram type

"""

from pathlib import Path
import numpy as np
import pandas as pd
import math
import sys
import warnings
from osgeo import gdal as gdal
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import skgstat as skg


import geoutils.raster as raster
from dbflow.src.db_utility import connect2db, create_sql, query_sql

import logging
import config as cfg


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)




def aggregate_raster(input_raster, output_raster, aggregation_factor):
    """
    Aggregate all bands of a raster dataset to a lower resolution while maintaining the same image size.

    This function reads a raster dataset, aggregates its bands by the specified factor, and saves the result
    to an output raster file. Each pixel in the output raster represents the mean value of the corresponding
    aggregated window from the input raster.

    Parameters
    ----------
    input_raster : str
        Path to the input raster dataset.

    output_raster : str
        Path to the output aggregated raster dataset.

    aggregation_factor : int
        Factor by which the resolution is reduced. A factor of 2 aggregates 2x2 windows of pixels.

    Returns
    -------
    None
        The aggregated raster is saved to the specified output path.

    Raises
    ------
    FileNotFoundError
        If the input raster file cannot be found.

    ValueError
        If the aggregation factor is not a positive integer or if the input raster cannot be read.

    Examples
    --------
    Aggregate a raster dataset by a factor of 4 and save the result:

    >>> aggregate_raster('data/input.tif', 'data/output_aggregated.tif', 4)
    """
    # Load raster dataset
    ifile = gdal.Open(input_raster)
    if ifile is None:
        raise FileNotFoundError(f"Input raster file '{input_raster}' not found or could not be opened.")

    if not isinstance(aggregation_factor, int) or aggregation_factor <= 0:
        raise ValueError("Aggregation factor must be a positive integer.")

    num_bands = ifile.RasterCount

    # Get the geotransform and projection
    geotransform = ifile.GetGeoTransform()
    projection = ifile.GetProjection()

    # Create an empty list to hold aggregated arrays for each band
    aggregated_arrays = []

    # Iterate over each band
    for band_number in range(1, num_bands + 1):
        band = ifile.GetRasterBand(band_number)
        array = band.ReadAsArray()

        # Calculate the dimensions of the output raster
        output_height = array.shape[0] // aggregation_factor
        output_width = array.shape[1] // aggregation_factor

        # Create an empty aggregated array for this band
        aggregated_array = np.zeros((output_height, output_width))

        # Aggregate pixels for this band
        for i in range(output_height):
            for j in range(output_width):
                window = array[i * aggregation_factor:(i + 1) * aggregation_factor,
                               j * aggregation_factor:(j + 1) * aggregation_factor]
                aggregated_array[i, j] = np.mean(window)

        # Append aggregated array for this band to the list
        aggregated_arrays.append(aggregated_array)

    # Adjust the geotransform parameters for the output raster
    new_geotransform = list(geotransform)
    new_geotransform[1] *= aggregation_factor
    new_geotransform[5] *= aggregation_factor

    # Save aggregated arrays as raster
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_raster, output_width, output_height, num_bands, gdal.GDT_Float32)
    outdata.SetGeoTransform(new_geotransform)
    outdata.SetProjection(projection)
    for i, aggregated_array in enumerate(aggregated_arrays, start=1):
        outband = outdata.GetRasterBand(i)
        outband.WriteArray(aggregated_array)
    outdata.FlushCache()


def get_info_from_db(sql_file, db_engine, replacements=None):
    """
    Execute an SQL query from a file with optional dynamic parameters and return the results as a pandas DataFrame.

    This function reads an SQL query from a file and optionally replaces placeholders with values
    provided in the `replacements` dictionary. The query is then executed to retrieve data from the database.

    Parameters
    ----------
    sql_file : str
        Path to the SQL file containing the query template. This parameter is required.
        The SQL file must exist and contain a valid SQL query.

    replacements : dict, optional
        A dictionary containing dynamic parameters to be inserted into the SQL query.
        If not provided, the SQL query will run as is without any parameter substitution.
    db: db.archive

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the results of the executed SQL query.

    Raises
    ------
    FileNotFoundError
        If the specified SQL file cannot be found.
    ValueError
        If the SQL file is empty or invalid.

    Examples
    --------
    >>> replacements = {'start_date': '2023-01-01', 'end_date': '2023-12-31'}
    >>> df = get_info_from_db('queries/get_data.sql', dbarchive.archive.engine, replacements)
    >>> print(df.head())

    >>> df_no_params = get_info_from_db('queries/simple_query.sql')
    >>> print(df_no_params.head())
    """
    if replacements is None:
        replacements = {}

    sql = create_sql(sql_file=sql_file, replacements=replacements)
    df = query_sql(sql, db_engine)
    return df


def manual_bbch_assignment(ipath_stem):
    """
    Manual BBCH code assignment based on expert knowledge and UAV image interpretation
    """

    manual_bbch_dict = {
        'MRKN_A2060-00_20210428_SG': '0',
        'FRIEN_90-00_20190704_SG': '87',
        'DEMM_3332_20210613_WW': '55',
        'DEMM_3371_20210509_WW': '33',
        'MRKN_A2060_20210601_SG': '28',
        'MRKN_A2060_20210721_SG': '85',
        'MRKN_A3200_20210721_WR': '83',
        'MRKN_A3100_20210428_WR': '22',
        'MRKN_A3100_20210428_WR_shadow': '22',
        'MRKN_A6090_20210721_KM': '37',
        'MRKN_A6070-00_20210721_KM': '27',
        'MRKN_A6070_20210927_KM': '87',
        'MRKN_A6090_20210927_KM': '87',
        'FRIEN_93_20210703_WW_vitalLodgedCrop': '80',
        'MRKN_A2060_20210721_SG_vitalLodgedArea': '85',
        'MRKN_A1270_20210820_SG_dryLodgedCrop': '97',
        'MRKN_A1270-00_20210820_SG': '87'
    }
    return manual_bbch_dict.get(ipath_stem, 'Unknown')


def extract_bbch(db, formatted_date, fid, crop_type_code, ipath_stem):
    """
    Retrieve the BBCH growth stage for a specific crop, with fallback to manual assignment if no data is available.

    This function acts as a wrapper around `get_info_from_db`, querying the BBCH growth stage
    for a crop based on the provided parameters. If no BBCH information is found in the database,
    the function assigns a value manually and logs the process.

    Parameters
    ----------
    formatted_date : str
        The date formatted to query the database for BBCH values.

    fid : str
        A field identifier used to locate the crop record.

    crop_type_code : str
        The code representing the crop type.

    ipath_stem : str
        Path stem or identifier used for manual BBCH assignment if no data is available from the database.

    Returns
    -------
    float or int
        The BBCH growth stage, either retrieved from the database (median value) or assigned manually.

    Raises
    ------
    FileNotFoundError
        If the SQL file `_query_bbch.sql` is not found.
    ValueError
        If the query returns invalid or empty data, and manual assignment also fails.

    Examples
    --------
    Query BBCH for a specific crop and date:

    >>> extract_bbch('2023-05-15', '63-00', 'WW', 'path/to/image') #  'YYYY-MM-DD' format
    No BBCH information is in DB available! 45 was assigned manually
    45
    """
    logger.info("Starting BBCH extraction for fid: %s, crop_type_code: %s, date: %s", fid, crop_type_code,
                formatted_date)

    replacements = {
        ':formatted_date': formatted_date,
        ':fid': fid,
        ':crop_type_code': crop_type_code
    }

    try:
        df = get_info_from_db(db_engine=db.archive.engine, replacements=replacements, sql_file='_query_bbch.sql')
        logger.info("BBCH query executed successfully. Retrieved %d rows.", len(df))
    except FileNotFoundError as e:
        logger.error("SQL file not found: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error during BBCH extraction.")
        raise

    if df['bbch'].isna().all():
        bbch = manual_bbch_assignment(ipath_stem)
        logger.warning("No BBCH information in DB. Assigned manually: %s", bbch)
        print('No BBCH information is in DB available! {} was assigned manually'.format(bbch))
    else:
        bbch = df['bbch'].median()
        logger.info("BBCH value retrieved from DB: %s", bbch)

    return bbch


def process_hsi(ifile, ipath, x_offset, y_offset, min_size, selection=[3, 2, 1], color_space=['I']):
    """
    Process HSI (Hue, Saturation, Intensity) transformation on a raster file.

    Parameters
    ----------
    ifile : raster.Raster
        Raster object representing the input image.
    ipath : pathlib.Path
        Path to save the transformed HSI raster.
    x_offset : int
        X offset to read from the raster.
    y_offset : int
        Y offset to read from the raster.
    min_size : int
        Minimum size of the subset to be processed.
    selection : list of int, optional
        List of band indices to be used for HSI conversion (default is [3, 2, 1]).
    color_space : list of str, optional
        List of color space channels to process (default is ['I']).

    Returns
    -------
    dict
        Dictionary of processed channels with channel names as keys and arrays as values.
    """
    array_set = ifile.RGB2HSI(selection)

    ds_dict = {}
    for ch in color_space:
        output_transform = ipath.parent.joinpath(f"{ipath.stem}{ch}.tif").as_posix()

        ds_transform = ifile.create_ds(output_filename=output_transform,
                                       bnumber=1,
                                       fformat='GTiff',
                                       datatype=gdal.GDT_Float32)

        ifile.write(output_filename=output_transform,
                    new_ds=ds_transform,
                    array=array_set[ch],
                    nan=None,
                    driver_format='GTiff')
        ds_transform = None
        # read created hsi raster again
        ifile = raster.Raster(output_transform)
        ds_dict[ch] = ifile.ds.GetRasterBand(1).ReadAsArray(x_offset, y_offset, min_size, min_size)

    return ds_dict


def process_lbp(ifile, ipath, x_offset, y_offset, min_size, selection=[3, 2, 1], p=4,r=3, layer_name=['LBP']):
    """
    Process Local Binary Pattern (LBP) analysis on a raster file.

    Parameters
    ----------
    ifile : raster.Raster
        Raster object representing the input image.
    ipath : pathlib.Path
        Path to save the LBP raster.
    x_offset : int
        X offset to read from the raster.
    y_offset : int
        Y offset to read from the raster.
    min_size : int
        Minimum size of the subset to be processed.
    selection : list of int, optional
        List of band indices for LBP calculation (default is [3, 2, 1]).
    p : int, optional
        Number of circularly symmetric neighbor set points (default is 4).
    r : int, optional
        Radius of the circle (default is 3).
    layer_name : list of str, optional
        Name of the output layer (default is ['LBP']).

    Returns
    -------
    dict
        Dictionary containing processed LBP arrays.
    """
    ds_dict = {}
    for ch in layer_name:

        output_transform = ipath.parent.joinpath(f"{ipath.stem}p{p}_r{r}_{ch}.tif").as_posix()

        lbp = raster.Raster.calculate_lbp(ifile, p=p, r=r, band_number=selection)

        ds_transform = ifile.create_ds(output_filename=output_transform,
                                       bnumber=1,
                                       fformat='GTiff',
                                       datatype=gdal.GDT_Float32)

        ifile.write(output_filename=output_transform,
                    new_ds=ds_transform,
                    array=lbp,
                    nan=None,
                    driver_format='GTiff')
        ds_transform = None
        # read created lbp raster again
        ifile = raster.Raster(output_transform)
        ds_dict[ch] = ifile.ds.GetRasterBand(1).ReadAsArray(x_offset, y_offset, min_size, min_size)

    return ds_dict


def process_glcm(ifile, ipath, x_offset, y_offset, min_size, selection=[3, 2, 1], layer_name=['GLCM']):
    """
    Process Gray Level Co-occurrence Matrix (GLCM) analysis on a raster file.

    Parameters
    ----------
    ifile : raster.Raster
        Raster object representing the input image.
    ipath : pathlib.Path
        Path to save the GLCM raster.
    x_offset : int
        X offset to read from the raster.
    y_offset : int
        Y offset to read from the raster.
    min_size : int
        Minimum size of the subset to be processed.
    selection : list of int, optional
        List of band indices for GLCM calculation (default is [3, 2, 1]).
    layer_name : list of str, optional
        Name of the output layer (default is ['GLCM']).

    Returns
    -------
    dict
        Dictionary containing processed GLCM arrays.
    """
    ds_dict = {}
    for ch in layer_name:

        output_transform = ipath.parent.joinpath(f"{ipath.stem}_{ch}.tif").as_posix()

        glcm = raster.Raster.calculate_GLCM(ifile, band_number=selection)

        ds_transform = ifile.create_ds(output_filename=output_transform,
                                       bnumber=1,
                                       fformat='GTiff',
                                       datatype=gdal.GDT_Int16)

        ifile.write(output_filename=output_transform,
                    new_ds=ds_transform,
                    array=glcm,
                    nan=None,
                    driver_format='GTiff')
        ds_transform = None
        # read created glcm raster again
        ifile = raster.Raster(output_transform)
        ds_dict[ch] = ifile.ds.GetRasterBand(1).ReadAsArray(x_offset, y_offset, min_size, min_size)

    return ds_dict



def process_uav_channels(ifile, x_offset, y_offset, min_size, layer_name = ['B', 'G', 'R']):
    """
    Customized function for Red channel (RGB) and Red Edge Channel (RGB,RE,NIR)
    """
    ds_dict = {}
    for i, ch in enumerate(layer_name):
        ds_dict[ch] = ifile.ds.GetRasterBand(i + 1).ReadAsArray(x_offset, y_offset, min_size, min_size)

    return ds_dict


def process_raster(calculate, ifile, ipath, x_offset, y_offset, min_size):
    """
    Process raster data based on the specified calculation method.

    Parameters
    ----------
    calculate : str or list
        Calculation method ('HSI', 'LBP', 'GLCM') or list of channels.
    ifile : raster.Raster
        Raster object representing the input image.
    ipath : pathlib.Path
        Path to save the output raster.
    x_offset : int
        X offset to read from the raster.
    y_offset : int
        Y offset to read from the raster.
    min_size : int
        Minimum size of the subset to be processed.

    Returns
    -------
    dict
        Dictionary containing processed raster arrays.
    """

    if calculate == 'HSI':
        ds_dict = process_hsi(ifile, ipath, x_offset, y_offset, min_size)

        return ds_dict

    elif calculate == 'LBP':
        ds_dict = process_lbp(ifile, ipath, x_offset, y_offset, min_size)

        return ds_dict

    elif calculate == 'GLCM':
        ds_dict = process_glcm(ifile, ipath, x_offset, y_offset, min_size)

        return ds_dict

    else:
        if isinstance(calculate, list):
            ds_dict = process_uav_channels(ifile, x_offset, y_offset, min_size, layer_name=calculate)

            return ds_dict

        elif isinstance(calculate, str):
            ds_dict = process_uav_channels(ifile, x_offset, y_offset, min_size, layer_name=[calculate])

            return ds_dict
        else:
            print("calculate is neither a list nor a string")


def create_grid_coordinates(ifile):
    """
    Generate grid coordinates for a raster file.

    This function calculates the x and y grid coordinates based on the raster's
    spatial resolution and extent. It returns arrays representing the x and y
    coordinates of the raster grid.

    Parameters
    ----------
    ifile : Raster
        A Raster object that contains metadata such as minimum and maximum x/y coordinates
        and the resolution of the raster (xres and yres).

    Returns
    -------
    tuple of numpy.ndarray
        - x (numpy.ndarray): 1D array of x-coordinates for the raster grid.
        - y (numpy.ndarray): 1D array of y-coordinates for the raster grid.

    Notes
    -----
    - The x coordinates are generated from the minimum x (minx) to the maximum x (maxx) using the raster's resolution (xres).
    - The y coordinates are generated from the minimum y (miny) to the maximum y (maxy) using the raster's resolution (yres).
    - This function assumes the raster uses regular grid spacing.

    Examples
    --------
    Generate grid coordinates for a raster object:

    >>> raster_file = raster.Raster('input_data.tif')
    >>> x, y = create_grid_coordinates(raster_file)
    >>> print(x.shape, y.shape)
    (512,) (512,)
    """
    x = np.arange(ifile.minx, ifile.maxx, abs(ifile.xres))
    y = np.arange(ifile.miny, ifile.maxy, abs(ifile.yres))
    return x, y


def collect_pixel_coordinates(x, y):
    """
    Generate pixel coordinates from grid dimensions.

    This function iterates over the provided x and y grid coordinates and collects
    all possible pixel coordinates in the form of (x, y) pairs.

    Parameters
    ----------
    x : numpy.ndarray
        1D array of x grid coordinates.

    y : numpy.ndarray
        1D array of y grid coordinates.

    Returns
    -------
    list of list
        A list containing pixel coordinates, where each element is a list `[xx, yy]`
        representing the indices of the pixel in the grid.

    Notes
    -----
    - The function generates a grid of size len(x) by len(y).
    - Pixel coordinates are collected in row-major order (x changes faster than y).
    - This function does not account for spatial referencing, only the pixel grid indices.
    """
    coords_pixel = []
    for xx in range(len(x)):
        for yy in range(len(y)):
            coords_pixel.append([xx, yy])
    return coords_pixel


def setup_plot(channel_data, subset_size):
    """
    Set up a plot to visualize raster channel data.

    This function creates a plot displaying a subset of the raster channel data using
    a specified colormap. The subset size defines the portion of the raster to be plotted.

    Parameters
    ----------
    channel_data : numpy.ndarray
        2D array representing the raster channel data to be visualized.

    subset_size : int
        Size of the subset to be displayed in pixels. The plot will visualize the
        top-left `subset_size x subset_size` region of the raster.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object of the plot, allowing for further customization.

    Notes
    -----
    - The plot uses the 'viridis' colormap for visualization.
    - Aspect ratio is set to 'equal' to maintain the correct grid proportions.
    - The function assumes global variables `bbch` and `date` are defined externally
      to format the plot's x-label.

    """
    plt.figure()
    plt.imshow(channel_data[:subset_size, :subset_size], cmap='viridis')
    ax = plt.gca()
    ax.set_aspect('equal')
    x_label = ax.set_xlabel('BBCH - {}, {}'.format(bbch, datetime.strptime(date, '%Y%m%d').date()),
                            labelpad=17, fontsize=26)
    return ax


def configure_ticks(ax, chess_size, subset_size):
    """
    Configure the ticks and grid for the plot based on the chessboard pattern size.

    This function sets up the major and minor ticks for the x and y axes of the plot.
    It applies a grid based on the minor ticks to create a chessboard-like appearance,
    adjusts tick labels, and places the x-axis label at the bottom of the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object of the plot to configure.

    chess_size : int
        Size of one square in the chessboard pattern (in pixels).

    subset_size : int
        Size of the subset to be displayed (in pixels).

    Notes
    -----
    - Major and minor ticks are both based on `chess_size`.
    - Gridlines are applied at both major and minor ticks.
    - Tick labels and font sizes are customized for better visibility.
    - The x-axis label is formatted using the global `bbch` and `date` variables.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> configure_ticks(ax, chess_size=30, subset_size=300)
    >>> plt.show()
    """
    ticks = np.arange(0, subset_size, chess_size)

    # Major ticks
    # ax.set_xticks(np.arange(0, ifile.xsize, abs(ifile.xres) * 900 * 10))
    # ax.set_yticks(np.arange(0, ifile.ysize, abs(ifile.yres) * 900 * 10))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Labels for major ticks
    # ax.set_xticklabels(np.arange(0, ifile.xsize, round(abs(ifile.xres) * 900 * 10)))
    # ax.set_yticklabels(np.arange(0, ifile.ysize, round(abs(ifile.yres) * 900 * 10)))


    # Minor ticks
    # ax.set_xticks(np.arange(0, ifile.xsize, abs(ifile.xres) * 900), minor=True)
    # ax.set_yticks(np.arange(0, ifile.ysize, abs(ifile.yres) * 900), minor=True)
    ax.set_xticklabels(ticks, minor=True)
    ax.set_yticklabels(ticks, minor=True)

    # Gridlines based on minor ticks
    ax.grid(visible=True, which='both', color='w', linestyle='-', linewidth=2)
    ax.tick_params(axis='both', labelsize=26)


    # Set the locator_params to skip every other tick
    #skip_ticks = chess * 2  # Skip every other tick

    #ax.xaxis.set_major_locator(plt.MultipleLocator(skip_ticks))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(skip_ticks))

    #ax.grid(visible=True, axis='both', which='major', color='w', linestyle='-', linewidth=2)
    ax.set_frame_on(False)

    # want a more natural, table-like display
    # ax.invert_yaxis()
    ax.xaxis.tick_top()
    # to place the tick marks at the top of the image
    x_label = ax.set_xlabel('BBCH - {}, {}'.format(bbch, datetime.strptime(date, '%Y%m%d').date(), ), labelpad=17, fontsize=26)
    # Set the color of the labels to black
    x_label.set_color('black')

    # Access the x and y axis objects
    x_axis = plt.gca().xaxis
    y_axis = plt.gca().yaxis
    # Set the font size for x and y tick labels
    font_size = 28


    # Change the tick label color to black
    for tick_label in x_axis.get_ticklabels():
        tick_label.set_fontsize(font_size)
        tick_label.set_color('black')

    for tick_label in y_axis.get_ticklabels():
        tick_label.set_fontsize(font_size)
        tick_label.set_color('black')

    ax.xaxis.set_label_position('bottom')


def save_plot(ax, output_path, filename):
    """
    Save the plot to the specified directory with the given filename.

    This function saves the plot associated with the provided axes object (`ax`) to a `_png`
    subdirectory of the specified `output_path`. The function ensures the directory exists,
    applies a tight layout for better formatting, and saves the plot in PNG format.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object representing the plot to be saved.

    output_path : pathlib.Path
        The base path where the `_png` subdirectory will be created, and the plot will be saved.

    filename : str
        The filename to save the plot under, including the `.png` extension.

    Notes
    -----
    - A `_png` subdirectory is automatically created if it does not exist.
    - The function applies padding to ensure the layout fits well within the saved image.
    - After saving, the plot is closed to free memory.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 1], [0, 1])
    >>> save_plot(ax, pathlib.Path('output'), 'example_plot.png')
    """

    #date_str = datetime.now().strftime('%Y%m%d')
    #filename = f'{channel}_{unit}_{pattern}_{bbch}_{date_str}.png'
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    output_dir = output_path.parent.joinpath('_png')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir.joinpath(filename).as_posix())
    plt.close()


def main_plot_function(channel, chess_size, output_path, filename):
    """
    Create and save a plot of raster data with a chessboard-like grid overlay.

    This function generates a plot for raster data, applies a grid pattern based on the provided
    chess size, and saves the plot to the specified path with the given filename.

    Parameters
    ----------
    channel : numpy.ndarray
        2D array representing raster channel data to be visualized.

    chess_size : int
        Size of one square in the chessboard grid pattern, in pixels.

    output_path : pathlib.Path
        Path where the output plot will be saved. The `_png` subdirectory will be created
        if it does not exist.

    filename : str
        The filename under which the plot will be saved, including the `.png` extension.

    Notes
    -----
    - The function sets the subset size to 5 times the chess size plus a padding of 2 pixels.
    - Ticks and gridlines are configured based on the chess size to ensure alignment with the plot.
    - The final plot is saved as a PNG in the `_png` subdirectory of the output path.
    """
    subset_size = (chess_size * 5) + 2
    ax = setup_plot(channel, subset_size)
    configure_ticks(ax, chess_size, subset_size)
    print(output_path)
    print(filename)
    save_plot(ax, output_path, filename)


def plot_variogram(V, model, estimator, unit, plot_dir, plot_filename):
    """
    Plot the experimental variogram and fitted model, and save the resulting plot to a file.

    This function visualizes the experimental variogram by plotting the experimental points
    and overlaying the fitted model curve. The plot is saved in the specified directory with
    the provided filename.

    Parameters
    ----------
    V : object
        Variogram object that contains `bins`, `experimental`, and `fitted_model` attributes,
        as well as `describe()` and `rmse` methods.

    model : str
        Name of the model used to fit the variogram (e.g., 'spherical', 'exponential').

    estimator : str
        Estimator used to calculate the variogram (e.g., 'ordinary', 'robust').

    unit : str
        Measurement unit for the semi-variance (e.g., 'DN' or 'reflectance').

    plot_dir : pathlib.Path
        Path to the directory where the plot will be saved.

    plot_filename : str
        Filename for saving the plot, including the `.png` or `.jpg` extension.

    Notes
    -----
    - The function plots the experimental variogram as scatter points and the fitted model
      as a dashed curve.
    - The plot title includes the model name, estimator, and the RMSE value of the fitted model.
    - The `effective_range` is printed to the console for further inspection.
    """
    # Create figure and axis
    fig, axs = plt.subplots(1, 1, figsize=(12, 6))

    # Plot the experimental variogram
    axs.scatter(V.bins, V.experimental, label='Experimental Variogram', color='blue')

    # Plot the fitted curve
    lags = np.linspace(0, np.max(V.bins), 100)  # Generate 100 points between 0 and the maximum lag
    model_values = V.fitted_model(lags)  # Evaluate the fitted model at those points
    axs.plot(lags, model_values, label='Fitted Model', color='magenta', linestyle='--')

    # Retrieve effective range and print it
    r_exp = V.describe().get('effective_range')
    print('Effective range:', r_exp)

    # Set title and labels
    axs.set_title(f"Variogram (Model: {model}, Estimator: {estimator}, RMSE: {V.rmse})", fontsize=20)
    axs.set_xlabel('Lag [pixel]', fontsize=26, labelpad=15)
    axs.set_ylabel(f'Semi-variance [{unit}]', fontsize=26, labelpad=15)
    axs.tick_params(axis='both', labelsize=26)

    # Save the plot
    plot_path = plot_dir / plot_filename
    plt.savefig(plot_path)

    # Close the plot to prevent memory leaks
    plt.close(fig)

    # Print progress
    print(f"Plot saved: {plot_path}")


def insert_variogram_data_to_db(dbarchive,
                          model, estimator, rmse, min_size, number_ploted_pixels,
                          ipath, crop_type_code, bbch, pattern,
                          uav_date, fid, aoi, channel, unit,
                          best_model, best_estimator, best_rmse,
                          insert2db=True):
    """
    Insert variogram model performance data into a database.

    This function prepares variogram performance metrics and metadata, and inserts the data
    into the `uavsemivarmodelperform` table in the database. The insertion can be controlled
    using the `insert2db` flag.

    Parameters
    ----------
    dbarchive : DatabaseArchive
        Database access object responsible for handling database operations.

    model : str
        Name of the model used to fit the variogram (e.g., 'spherical', 'exponential').

    estimator : str
        Estimator method applied to calculate the variogram (e.g., 'ordinary', 'robust').

    rmse : float
        Root Mean Square Error of the fitted variogram model.

    min_size : int
        Minimum size of the raster subset in pixels.

    number_ploted_pixels : int
        Number of pixels plotted in the variogram visualization.

    ipath : str
        File path to the subset or raster data used in the variogram analysis.

    crop_type_code : str
        Code representing the crop type associated with the variogram data.

    bbch : str
        BBCH scale value representing the phenological growth stage.

    pattern : str
        Description of the spatial pattern or data aggregation method.

    uav_date : str
        UAV imaging date in the format 'YYYY-MM-DD'.

    fid : int
        Feature ID associated with the data subset.

    aoi : str
        Area of interest identifier or description.

    channel : str
        Data channel used (e.g., 'R', 'G', 'B', 'NIR').

    unit : str
        Unit of measurement for the data (e.g., 'DN' or 'reflectance').

    best_model : str
        Name of the best-performing model based on evaluation metrics.

    best_estimator : str
        Estimator method associated with the best-performing model.

    best_rmse : float
        RMSE value of the best-performing model.

    insert2db : bool, optional
        If `True`, the data is inserted into the database (default is True).

    Notes
    -----
    - The data is inserted into the `uavsemivarmodelperform` table.
    - If `insert2db` is `False`, the function prepares the data but does not insert it.
    - After successful insertion, `new_data` is optionally set to `None` to clear memory.

    """
    new_data = {
        'model': model,
        'estimator': estimator,
        'rmse': rmse,
        'subset_size_pixels': min_size,
        'number_ploted_pixels': number_ploted_pixels,
        'subset': ipath,
        'crop_type_code': crop_type_code,
        'bbch': bbch,
        'pattern': pattern,
        'uav_date': uav_date,
        'fid': fid,
        'aoi': aoi,
        'channel': channel,
        'unit': unit,
        'best_model': best_model,
        'best_estimator': best_estimator,
        'best_rmse': best_rmse
    }

    if insert2db:
        dbarchive.insert(table='uavsemivarmodelperform',
                         primary_key=dbarchive.get_primary_keys('uavsemivarmodelperform'),
                         orderly_data=[new_data],
                         update=True
                         )
        new_data = None  # Optionally reset new_data after insertion


def check_semivariogram_size(min_size, number_ploted_pixels):
    """
    Check the size of the semivariogram relative to the image and issue warnings if necessary.

    Parameters
    ----------
    min_size : int
        Minimum size of the image subset in pixels.
    number_ploted_pixels : int
        Number of plotted pixels in the semivariogram.
    """
    ratio = round(min_size / number_ploted_pixels)

    if ratio < 5:
        message = (f"Semivariogram is plotted for more than 1/5 of the image size: "
                   f"{number_ploted_pixels} - {min_size} - {ratio}!")
        warnings.warn(message, UserWarning)
        logging.warning(message)

    if ratio < 2:
        message = (f"Semivariogram is plotted for more than 1/2 of the image size: "
                   f"{number_ploted_pixels} - {min_size} - {ratio}!")
        warnings.warn(message, UserWarning)
        logging.warning(message)
        sys.exit("Exiting due to large semivariogram size.")  # Graceful exit



def read_config(file_path):
    """
    Reads a configuration file with section headers and returns a dictionary of parameters.

    Parameters
    ----------
    file_path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Dictionary with keys as section headers and values as lists of parameters in correct order.
    """
    input_parameters = {}
    current_section = None
    temp_params = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines or comments
            if not line or line.startswith('#'):
                continue

            # Detect section headers like [14]
            if line.startswith('[') and line.endswith(']'):
                if current_section and temp_params:
                    # Reorder and store the previous section
                    input_parameters[current_section] = [
                        temp_params.get('path', ''),
                        temp_params.get('sub_class_name', ''),
                        temp_params.get('plot_legend', False)
                    ]
                current_section = line[1:-1]
                temp_params = {}
                continue

            # Ensure section exists before parsing parameters
            if current_section is None:
                print(f"Skipping line (no section): {line}")
                continue

            # Parse key-value pairs
            try:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Handle lists or boolean conversion
                if value.startswith('[') and value.endswith(']'):
                    value = eval(value)
                    value = value[0] if isinstance(value, list) else value
                else:
                    value = value.replace('\\', '/')

                # Store parameters temporarily to reorder later
                temp_params[key] = value

            except ValueError as e:
                print(f"Skipping invalid line: {line} - Error: {e}")

        # Store the last section
        if current_section and temp_params:
            input_parameters[current_section] = [
                temp_params.get('path', ''),
                temp_params.get('sub_class_name', ''),
                temp_params.get('plot_legend', False)
            ]

    return input_parameters







# Set the global font of tehplots to Arial
plt.rcParams['font.family'] = 'Arial'
pd.set_option("display.max_rows", None)  # Display all rows
pd.set_option("display.max_columns", None)
plt.style.use('ggplot')



#---------------- SET PARAMETERS

# use this to plot the legend of the plots on the top of the plot fo GSD=0.027
set_ylim = False

#legend_location = 'upper right'
legend_location = 'lower right'

insert2db = False
agregate = False
plot_semivariogram = True

# define the output directory
output_dir = Path(r'...\1_UAV_Paper\_tif')



# ---------------- Read the input paths of UAV subset Images for analysis

# Generate a file in texture_analysis/input_paths.txt with Links to a file to be read
# Example
"""
[x]
sub_class_name = ['vital_crop']
plot_legend = [False]
path = ..._uav_paper\_plots\semivar_script_test\MRKN_A2060_20210601_SG.tif


[0]
sub_class_name = ['vital_crop']
plot_legend = [False]
path = ..._uav_paper\_plots\UAV_subsets\MRKN_A6070-00_20210721_KM_vital_crop.tif
"""

# Get the directory of the current script
current_dir = Path(__file__).parent

# Construct the path to the input file
config_path = current_dir / 'input_paths.txt'
input_parameters = read_config(config_path)


# Print to verify
for key, value in input_parameters.items():
    print(f"{key}: {value}")



db_path_main = cfg.db_path
dbarchive = connect2db(db_path_main)  # the info about BBCH code that stored in DB to be queried


# if this tables do not exist in Databank create them using txt file 'create_....txt'
print(dbarchive.get_primary_keys('uavsemivariogram'))
print(dbarchive.get_colnames('uavsemivariogram'))

# and also the results will be inserted into table 'uavsemivarmodelperform' in RCM.db
print(dbarchive.get_primary_keys('uavsemivarmodelperform'))
print(dbarchive.get_colnames('uavsemivarmodelperform'))




# -----------------------------------------------------------------------------------------------------------
if agregate == True:
    for i, (ipath, PATTERN, plot_legend) in input_parameters.items():
        ipath = Path(ipath)

        for i in [6]:
            aggregate_raster(input_raster=ipath.as_posix(),
                             output_raster=ipath.parent.joinpath(ipath.stem + '_{}.tif'.format(i)).as_posix(),
                             aggregation_factor=i
                             )
            # for i in [2, 4, 8]:
            ifile = raster.Raster(ipath.parent.joinpath(ipath.stem + '_{}.tif'.format(i)).as_posix())
# -----------------------------------------------------------------------------------------------------------



for i, (ipath, PATTERN, plot_legend) in input_parameters.items():
    print(i, (ipath, PATTERN, plot_legend))

    ipath = Path(ipath)

    # split the file Header and assign variables
    if len(ipath.stem.split('_')) == 5:
        aoi, fid, date, crop_type_code, pattern = ipath.stem.split('_')
    elif len(ipath.stem.split('_')) == 4:
        aoi, fid, date, crop_type_code = ipath.name.split('_')
    elif len(ipath.stem.split('_')) == 6:
        aoi, fid, date, crop_type_code, xx, xx = ipath.stem.split('_')
    crop_type_code = crop_type_code.split('.')[0]


    formatted_date = f'{date[:4]}-{date[4:6]}-{date[6:]}' # Convert to 'YYYY-MM-DD' format

    ifile = raster.Raster(ipath.as_posix())
    bbch = extract_bbch(dbarchive, formatted_date, fid, crop_type_code, ipath.stem)


    # catch meta information of image to be inserted in DB
    CROP_TYPE_CODE = crop_type_code
    FID = fid
    BBCH = bbch
    UAV_DATE = date
    AOI = aoi


   # find the minimum side of the Raster (because UAV subsets are not equally from each side)
    min_size = min(ifile.xsize, ifile.ysize)
    if min_size > 1500:
        min_size = 1500

    # Calculate the starting pixel coordinates (x_offset, y_offset)
    x_offset = max(0, min(0, ifile.rows - min_size))  # Center the subset in the x direction
    y_offset = max(0, min(0, ifile.cols - min_size)) # Center the subset in the y direction



    if ifile.bnumber == 3:
        calculate = 'R'
    elif ifile.bnumber == 5:
        calculate = 'RE'


    #calculate = 'GLCM'
    ds_dict = process_raster(calculate, ifile, ipath, x_offset, y_offset, min_size)


    for i, (ch, ds) in enumerate(ds_dict.items()):

        CHANNEL = ds
        _CHANNEL = ch
        UNIT = 'DN' if date[:4] in ['2019', '2020'] else 'reflectance'
        custom_ticks_ = [0, 30, 60, 90, 120]

        print(f'CHANNEL: {_CHANNEL}, UNIT: {UNIT}, Ticks: {custom_ticks_}')

        wave_length = {   'B': 'Blue'
                        , 'G': 'Green'
                        , 'R': 'Red'
                        , 'NIR': 'Near-Infrared'
                        , 'RE': 'Red-Edge'
                        }


        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # ----------- Write the equally cropped array of the selected Channel on the Disk as Raster
        output_tif = output_dir.joinpath('{}_{}.tif'.format(ipath.stem, _CHANNEL))
        new_ds = ifile.create_ds(output_filename=output_tif.as_posix(),
                                 xsize=min_size,
                                 ysize=min_size,
                                  bnumber=1,
                                  fformat='GTiff',
                                 datatype=gdal.GDT_Float32
                                 )

        ifile.write(output_filename=output_tif.as_posix(),
                      new_ds=new_ds,
                      array=CHANNEL,
                      bnumber=ifile.bnumber,
                      nan=np.nan,
                      driver_format='GTiff')

        ifile.close()
        new_ds.FlushCache()
        CHANNEL = None
        B, G, R, RE, NIR = None, None, None, None, None


        # ------------ Read Raster again that was stored in a output folder
        ifile = raster.Raster(output_tif.as_posix())
        CHANNEL = ifile.ds.GetRasterBand(1).ReadAsArray(),

        # Remove the singleton dimension
        CHANNEL = np.squeeze(CHANNEL)

        x, y = create_grid_coordinates(ifile)
        coords_pixel = collect_pixel_coordinates(x, y)

        print('Raster resolution in coord. system:', abs(ifile.xres))
        pixels = round(abs(ifile.xsize / ifile.xres))
        print(pixels)


        chess = 15
        main_plot_function(channel=CHANNEL,
                           chess_size=chess,
                           output_path=output_dir,
                           #filename = output_dir.parent.joinpath('_png', ipath.stem + '_overview_{}_{}_{}_{}.png'.format(_CHANNEL, UNIT, PATTERN, bbch)).as_posix(),
                           filename=ipath.stem + '_overview_{}_{}_{}_{}.png'.format(_CHANNEL, UNIT, PATTERN, bbch)
                           )


    if plot_semivariogram == True:

        # Prepare data for variogram analysis
        values = CHANNEL.astype('float32').reshape(ifile.ysize * ifile.xsize)
        coords = np.array(coords_pixel)

        #sample_size = 650
        #n_lags = 650
        sample_size = round(math.sqrt(min_size ** 2 + min_size ** 2))
        n_lags = sample_size


        distance = 8 # in m distance to be considered
        number_ploted_pixels = round(distance / ifile.xres)

        print('Number of plotted pixels in semivariogram: {}'.format(number_ploted_pixels))



        replacements = {':subset_name': ipath.stem}
        df = get_info_from_db(db_engine=dbarchive.archive.engine, sql_file='_query_best_semivar_parameters.sql',  replacements=replacements)


        plot_dir = output_dir.parent.joinpath('_png')


        if not df.empty:
            model = df['model'].unique().item()
            estimator = df['estimator'].unique().item()
        else:
            print('The entry about the best model and estimator exist in DB ! Plot will not be ploted !')
            # Define a directory to save the plots
            #plot_dir = Path(r"F:\RCM\sarbian\xx_paper\1_UAV_Paper\_png\variogram_plots")
            #plot_dir = Path(r"F:\RCM\sarbian\xx_paper\1_UAV_Paper\_png")
            plot_dir.mkdir(parents=True, exist_ok=True)

            model_list = ['spherical', 'exponential', 'gaussian', 'cubic', 'stable', 'matern']
            estimators = ['matheron', 'cressie', 'dowd', 'minmax', 'entropy'] #'genton', - is excluded due to the calculation is very time consuming

            best_rmse = float('inf')
            best_model = None
            best_estimator = None


            #------------------ Loop through combinations of models and estimators and insert to DB
            for model in model_list:
                for estimator in estimators:
                    try:
                        # Create a variogram object
                        V = skg.Variogram(coords[:sample_size], values[:sample_size],
                                          normalize=True,
                                          use_nugget=False,
                                          model=model,
                                          n_lags=n_lags,
                                          maxlag=number_ploted_pixels,
                                          estimator=estimator)
                        rmse = V.rmse
                    except:
                        print('failed for : {} and {}'.format(model, estimator))
                        V = None
                        rmse = np.nan


                    if V is not None:
                        plot_variogram(V, model, estimator, UNIT, plot_dir,
                                            plot_filename = f"{ipath.stem}_{model}_{estimator}_rmse_{round(rmse, 9)}.png")

                    # Print some information for debugging
                    print("Model:", model)
                    print("Estimator:", estimator)
                    print(type(V))
                    print("RMSE:", rmse)
                    print('-' * 50)

                    placeholder=''
                    insert_variogram_data_to_db(dbarchive,
                                          model=model, estimator=estimator, rmse=rmse, min_size=min_size,
                                          number_ploted_pixels=number_ploted_pixels, ipath=ipath.stem,
                                          crop_type_code=CROP_TYPE_CODE, bbch=BBCH, pattern=PATTERN, uav_date=UAV_DATE,
                                          fid=FID, aoi=AOI, channel=_CHANNEL, unit=UNIT,
                                          best_model=placeholder,
                                          best_estimator=placeholder,
                                          best_rmse=placeholder,
                                          insert2db=True)

                    print(rmse)

                    if not (np.isnan(rmse)): # value number
                        if (rmse < best_rmse):# Check if this combination has the lowest RMSE so far
                            best_rmse = rmse
                            best_model = model
                            best_estimator = estimator
                            #_AOI = AOI
                            #_FID = FID
                            #_UAV_DATE = UAV_DATE
                            #_PATTERN = PATTERN
                            #_BBCH = BBCH
                            #_CROP_TYPE_CODE = CROP_TYPE_CODE

            #------------------ SELECTION of teh best model based on the RMSE value !!!

            # Print the best model and estimator
            print("Best Model:", best_model)
            print("Best Estimator:", best_estimator)
            print("Best RMSE:", best_rmse)


            placeholder = True
            insert_variogram_data_to_db(dbarchive,
                                          model=best_model, estimator=best_estimator, rmse=best_rmse, min_size=min_size,
                                          number_ploted_pixels=number_ploted_pixels, ipath=ipath.stem,
                                          crop_type_code=CROP_TYPE_CODE, bbch=BBCH, pattern=PATTERN, uav_date=UAV_DATE,
                                          fid=FID, aoi=AOI, channel=_CHANNEL, unit=UNIT,
                                          best_model=placeholder,
                                          best_estimator=placeholder,
                                          best_rmse=placeholder,
                                          insert2db=True)

            # query the best model and estimator again after update - use the same query as defined above
            #df = cfg.query_sql(sql, dbarchive)

        replacements = {':subset_name': ipath.stem}
        df = get_info_from_db(db_engine=dbarchive.archive.engine, sql_file='_query_best_semivar_parameters.sql',  replacements=replacements)



        if not df.empty:
            model = df['model'].unique().item()
            estimator = df['estimator'].unique().item()
        else:
            print(df)
            print('df is empty')
            sys.exit()

        print('best model and estimator: {} {}'.format(model, estimator))

        #model = 'gaussian'#
        model = 'exponential'
        estimator = 'dowd'
        #estimator = 'cressie'



        #model = 'matern' #use this only for shadow WR sample
        #estimator = 'dowd'#use this only for shadow WR sample


        #estimators = ['matheron', 'cressie', 'dowd', 'minmax', 'entropy']  # 'genton'

        if round(min_size / number_ploted_pixels) < 5: # TODO: make it as Warning, add logging
            check_semivariogram_size(number_ploted_pixels, min_size)
        if round(min_size / number_ploted_pixels) < 2:
            check_semivariogram_size(number_ploted_pixels, min_size)


        v_dict = {}
        v_plots = {}

        try:
            # Create a variogram object
            V = skg.Variogram(coords[:sample_size], values[:sample_size],
                              normalize=True,
                              use_nugget=False,
                              model=model,
                              n_lags=n_lags,
                              maxlag=number_ploted_pixels,
                              estimator=estimator)
            rmse = V.rmse
        except:

            print('failed for : {} and {}'.format(model, estimator))
            V = None

        if V is not None:
            plot_variogram(V, model, estimator, UNIT, plot_dir,
                               plot_filename=f"{ipath.stem}_{model}_{estimator}_rmse_{round(rmse, 9)}.png")



        DIR_MODEL = 'compass'  # argument 'bandwidth' is ignored if 'compass' is selected
        NORMALIZE = True

        DIRECTIONS = [0, 45, 90, 135]

        for az_angles in DIRECTIONS:
            if az_angles == 0:
                TOLERANCE = 180
            else:
                TOLERANCE = 90

            V = skg.DirectionalVariogram(coords[:sample_size], values[:sample_size] # for ww coords[100:sample_size+100], values[100:sample_size+100]
                                         , azimuth=az_angles
                                         , tolerance=TOLERANCE
                                         , bandwidth='q30'
                                         , normalize=NORMALIZE
                                         , use_nugget=False
                                         , model=model  # spherical, exponential, gaussian, cubic, stable, matern
                                         , n_lags=n_lags
                                         # 100 - 18 pixels; 200 - 9 pixels; 300 - 6 pixels; 1800 - 1 pixel each lag
                                         # , bin_func='even'
                                         , directional_model=DIR_MODEL
                                         , maxlag=number_ploted_pixels
                                         , estimator=estimator
                                         # 20   maximum of the lag should be not bigger than 1/5 of the transect length -> 300/5 = 60 -> maxlag
                                         )

            v_plots['az_{}'.format(az_angles)] = V
            v_dict['az_{}'.format(az_angles)] = V.describe()

        df = pd.DataFrame(v_dict)

        print(df.loc['effective_range', :])
        print(df.loc['effective_range', :].unique())
        print(df.loc['sill', :])
        print(df.loc['nugget', :])

        for k, v in df.loc['params', :].items():
            print(k, v)


        data_size = len(DIRECTIONS) + 1  # range 0..23
        viridis = plt.get_cmap('viridis')
        cnorm = colors.Normalize(vmin=0, vmax=data_size)
        scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=viridis)


        fix, ax = plt.subplots(1, 1, figsize=(14, 6))
        color_ = 'magenta'
        color_ = 'red'
        ax.axvline(x=df.loc['effective_range', :].min(), linestyle='--', color=color_)
        ax.axvline(x=df.loc['effective_range', :].max(), linestyle='--', color=color_)

        # Plot the fitted curve
        lags = np.linspace(0, np.max(V.bins), 100)  # Generate 100 points between 0 and the maximum lag
        model_values = V.fitted_model(lags)  # Evaluate the fitted model at those points
        #ax.plot(lags, model_values, label='Fitted Model', color='magenta', linestyle='--')


        for i, (k, V) in enumerate(v_plots.items()):
            print(k, V)
            if k == 'az_0':
                line1, = ax.plot(V.bins, V.experimental, color=scalarMap.to_rgba(i + 1), marker='>', alpha=0.6, linestyle='None',
                        markersize=9, label=k)
            elif k == 'az_90':  # this is to make colors visible on the plot
                line2, = ax.plot(V.bins, V.experimental, color=scalarMap.to_rgba(i + 2), marker='.', alpha=0.6, linestyle='None',
                        markersize=11, label=k)
            elif k == 'az_45':
                line3, = ax.plot(V.bins, V.experimental, color=scalarMap.to_rgba(i), marker='.', alpha=0.6, linestyle='None',
                                 markersize=11, label=k)
            elif k == 'az_135':
                line4, = ax.plot(V.bins, V.experimental, color=scalarMap.to_rgba(i), marker='.', alpha=0.6, linestyle='None',
                        markersize=11, label=k)


        #ax.axvline(x=1, linestyle='--')
        #ax.axvline(x=3, linestyle='--')


        ax.set_xlabel('lag [pixel]', fontsize=30, labelpad=15, color='black')
        #ax.set_ylabel('semi-variance [{}]'.format(UNIT), fontsize=28, labelpad=15)
        ax.set_ylabel('semi-variance', fontsize=30, labelpad=15, color='black')

        # Set the upper limit of y-axis
        if set_ylim == True:
            #ax.set_ylim(bottom=0.0000001, top=0.0000016)
            #ax.set_ylim(bottom=0.00000001, top=0.00000065)
            #ax.set_ylim(bottom=0.00000001, top=0.00000040)
            #ax.set_ylim(bottom=0.0000001, top=0.00000080)
            ax.set_ylim(bottom=0.0000001, top=0.00000032)

        ax.set_title('{}'.format(''))
        ax.tick_params(axis='both', labelsize=36, color ='black')

        y_label = ax.yaxis.get_label()
        # Set the font size of the y-axis label
        font_size = 34
        y_label.set_fontsize(font_size)

        # Access the x and y axis objects
        x_axis = plt.gca().xaxis
        y_axis = plt.gca().yaxis
        # Set the font size for x and y tick labels
        #font_size = 34

        # Change the tick label color to black
        for tick_label in x_axis.get_ticklabels():
            tick_label.set_fontsize(font_size)
            tick_label.set_color('black')

        for tick_label in y_axis.get_ticklabels():
            tick_label.set_fontsize(font_size)
            tick_label.set_color('black')

        if plot_legend == True:
            # Add a legend with customized marker size and other properties
            legend = plt.legend(loc=legend_location, fontsize=28, ncol=2)

            # Set the marker size in the legend
            for j, line in enumerate(legend.get_lines()):
                if j == 0:
                #if j == 1:
                    line.set_markersize(30)
                else:
                    line.set_markersize(50)  # Adjust the marker size value

        #plt.legend(loc='lower right', fontsize="xx-large", ncol=2)


        #custom_ticks = [0, 24, 48, 72, 96, 120]
        custom_ticks = custom_ticks_
        #custom_ticks = range(custom_ticks_[-1])
        #custom_labels = [str(tick) if tick in custom_ticks_ else '' for tick in custom_ticks]  # Custom tick labels

        # Set custom tick values on the x-axis
        #plt.xticks(custom_ticks, custom_labels)
        plt.xticks(custom_ticks)
        #plt.ylim(bottom=0, top=1e-5)  # Adjust the upper limit as needed
        ax.yaxis.offsetText.set_fontsize(34)


        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(output_dir.parent.joinpath('_png', ipath.stem + '_{}_{}_dir_variogram_{}_{}.png'.format(model,
                                                                                                           estimator,
                                                                                                           _CHANNEL,
                                                                                                           UNIT
                                                                                                           )).as_posix())

        plt.close()




       # Data to be entered into DB
        df = df.transpose()
        df['subset_size_pixels'] = min_size
        df['number_ploted_pixels'] = number_ploted_pixels
        df['subset'] = ipath.stem
        df['crop_type_code'] = CROP_TYPE_CODE
        df['bbch'] = BBCH
        df['pattern'] = PATTERN
        df['uav_date'] = UAV_DATE
        df['fid'] = FID
        df['aoi'] = AOI
        df['channel'] = _CHANNEL
        df['unit'] = UNIT
        df['image_resolution'] = ifile.xres
        df = df.drop(columns=['params', 'kwargs'])
        df = df.reset_index()
        df.rename(columns={'index': 'semivar_type'}, inplace=True)
        df = df.reset_index()
        df.rename(columns={'index': 'ind'}, inplace=True)

        column_to_drop = 'shape'
        # Drop the column if it exists in the DataFrame
        if column_to_drop in df.columns:
            df.drop(column_to_drop, axis=1, inplace=True)


        if insert2db == True:
            # insert the  data to a DB
            dbarchive.insert(table='uavsemivariogram',
                             primary_key=dbarchive.get_primary_keys('uavsemivariogram'),
                             orderly_data=df.to_dict(orient='records'),  update=True)




