'''
@author: Aranil

configuration file for rcm-dashboard
'''
from dbflow.src import db_utility as dbu
from decouple import config
from pathlib import Path
import os
import time
import logging
import logging.config
import sys





#-----------------------------------------------------
# DEFINE MAIN PATHS (LOCAL/SERVER)
#-----------------------------------------------------

# path to SQLite Database
db_path = config('DB_PATH')
output_path = config('OUTPUT_PATH')
log_path = config('LOGGING_CONFIG')
testfolder_path = config('TESTFOLDER_PATH')




path_rslv = os.path.split(os.path.dirname(os.path.abspath(log_path)))[1:]
fileName = Path(__file__).parent.parent / 'logging.config'
print(fileName)


# connect to db
dbarchive = dbu.connect2db(db_path)


crs_utm = {
              'DEMM': '32633'
            , 'FRIEN': '32632'
            , 'MRKN': '32633'
          }


#TODO: take logger from module agrisense!!!
def generate_logfile(directory=Path(__file__).parent.parent.joinpath('_logs'), logfilename=None):
    '''

    Parameters
    ----------
    directory: str
        directory to store logfiles
        if None: generates logfile in the directory where python script was compiled from
    logfilename: str
        logfile name with extension '.log'
        if None:  generates log-filename from the name of the python script and date when the script was compiled

    Returns
    -------
        str full path of the logfile

    '''
    if logfilename == None:
        # using sys.argv[0] logfile will be created only for a script which is executed and all logfile will be stored there
        logfilename = os.path.basename(sys.argv[0]).split('.')[0] + '_' + time.strftime("%Y%m%d%H%M%S", time.gmtime()) + '.log'
    if directory != None:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Folder '_logfiles' is already there")
        else:
            print("Folder '_logfiles' was created")
        log = directory.joinpath(logfilename).as_posix()
    else:
        log = logfilename
    return log



def get_logger(logfilename=os.path.basename(sys.argv[0]).replace('.py', '.log')):
    logging.config.fileConfig(fileName,
                              disable_existing_loggers=False,
                              defaults={'logfilename': generate_logfile(
                              logfilename=logfilename)},
                              )
    logger = logging.getLogger('file_logger')
    return logger




shp_buffer_dist = {
                  'DEMM': -30
                , 'FRIEN': -30
                #, 'MRKN': -20 #=> to get rid of the irrigation system and Tree shadows
                , 'MRKN': -10  # => to get rid of the irrigation system and Tree shadows
                #, 'FRIEN1': -30
                }




classes = [
                'bare_soil',
                'vital_crop',
                'vital_lodged_crop',
                'flowering_crop',
                'dry_crop',
                'dry_lodged_crop',
                'ripening_crop',
                'weed_infestation'
               ]


crop_types = {
                'WW': 'Winter Wheat',
                'KM': 'Corn',
                'SG': 'Spring Barley',
                'WR': 'Rapeseed',
                'WG': 'Winter Barley',
                }
