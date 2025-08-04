print("")
print(" *************************************************************** ")
print("")
print("                -= SHESE Python Catalyst ortho processing workflow =-")
print("                      Last modified 19072022 ")
print(" last modification for python 3.5 and CATALYST binaries")
print("***************************************************************")
print(" mosprep mosdef mosrun")

#
# This script runs the following PPFs in a top-down sequence and generates a
# ortho mosaic based on the input ortho images exported from Metashape
# earlier. All orthos must have the same georef and resolution.
# 1. mosrep - set hotspot  neighborhood and minsqdiff for most balanced results
# 2. mosdef
# 3. mosrun
# could be also combine with hotspot to correct the final mosaic again

import os, fnmatch
import pci
from pathlib import Path
from pci.fun import *
from pci.lut import *
from pci.fimport import fimport
from pci.pcimod import pcimod
from pci.model import model
from pci.exceptions import *
from pci.ortho import ortho
from pci.mosprep import mosprep
from pci.mosdef import mosdef
from pci.mosrun import mosrun
from pci.hotspot import hotspot
import logging





def run_mosprep( mfile,
                silfile,
                nodatval,
                normaliz, #="ADAPTIVE",  # HotSpot method often useful for airphoto imagery, adaptive for unregular patchy reflectance variation or ramp corrections
                balspec="NEIGHBORHOOD",  # neighborhood best for strong differences and strong variations between different overlapping images
                cutmthd="minsqdiff",
                sortmthd="NEARESTCENTER"
               ):
    '''
    '''
    return mosprep(mfile, silfile, nodatval, sortmthd, normaliz, balspec, "", "", [], cutmthd, mapunits)


def run_mosdef( silfile,
                mdfile,
                nodata,
                pxszout,
                dbic=[1,2,3,4,5],
                tispec="",
                tipostrn="",
                mapunits="UTM 32 U D000",
                blend=[25],
                ftype="PIX",
                foptions=""
                ):
    '''
    Create Mosaic Definition XML file within the ortho folder
    '''
    return mosdef(silfile, mdfile, dbic, tispec, tipostrn, mapunits, pxszout, blend, nodata, ftype, foptions)


def run_mosrun(
                silfile,
                mdfile,
                outdir,
                crsrcmap="",
                ):
    '''
    Create full resolution mosaic file
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return mosrun(silfile, mdfile, outdir, "", crsrcmap, "SKIP", "NONE", "", "BILIN")


def run_mosaic( mfile,
                silfile,
                mdfile,
                outdir,
                nodatval,
                balspec="NEIGHBORHOOD",
                dbic=[1,2,3,4,5], 
                mapunits="UTM 32 U D000",
                pxszout=[0.05],
                blend=[25],
                cutmthd="minsqdiff",
                sortmthd="NEARESTCENTER",
                normaliz="ADAPTIVE",# HotSpot method often useful for airphoto imagery, adaptive for unregular patchy reflectance
                ftype="PIX"
                ):

    run_mosprep(mfile,
                silfile,
                nodatval,
                normaliz=normaliz,  # HotSpot method often useful for airphoto imagery, adaptive for unregular patchy reflectance variation or ramp corrections
                balspec=balspec,
                cutmthd=cutmthd,
                sortmthd=sortmthd
                )

    ## Create Mosaic Definition XML file within the ortho folder
    run_mosdef(silfile,
                mdfile,
                nodata=nodatval,
                dbic=dbic,
                tispec="",
                tipostrn="",
                mapunits=mapunits,
                pxszout=pxszout,
                blend=blend,
                ftype=ftype,
                foptions="")
				
    ## Create full resolution mosaic file
    run_mosrun( silfile,
                mdfile,
                outdir
                )


def run_hotspot(fili="data1.pix", dbic=[1,2,3,4,5], filo=None):		
	# run hotspot
	dboc=dbic
	if filo == None:
                filo=fili
	else:
		filo
	return hotspot(fili, dbic, filo, dboc)
	
	
#if __name__ == '__main__':


# log everything to file
logging.basicConfig(filename='catalyst_mosaic.log', level=logging.DEBUG)


# create a logger
logger = logging.getLogger()
# configure a stream handler
logger.addHandler(logging.StreamHandler())
# log all messages, debug and up
logger.setLevel(logging.DEBUG)
	
    
    
				
TIF_PATTERN = 'tif'
NODATAVAL = [0] #  [-32767]
CHANNELS = [1,2,3,4,5] # [1,2,3]  # for rgb
MAPUNITS = "UTM 32U D000" # or "UTM 33U D000"
PXZOUT = [0.05] # [0.03] # for rgb
BLEND = [125] # BLEND=[25] # for rgb
CUTMTHD = "minsqdiff" # "EDGE"


#NORMALIZE = "ADAPTIVE" # HotSpot method often useful for airphoto imagery, adaptive for unregular patchy reflectance variation or ramp corrections
#BALSPEC = "NEIGHBORHOOD"  # neighborhood best for strong differences and strong variations between different overlapping images
#BALSPEC = 'HISTOGRAMM' / 'BUNDLE' only for images from the one path
#SORTMTHD = "NEARESTCENTER"



#root_dir = input("Insert Folder Path: ")
#decision1 = input("Local Mosaic (0/1): ")
#decision2 = input("Apply Hotspot (0/1): ")
#decision3 = input("Global Mosaic (0/1): ")

root_dir = input("Insert Folder Path: ")
decision1 = "0" # run mosaic within image strap
decision2 = "1" # hotspot
decision3 = "1" # to mosaic the straps to one image




def run_process(root_dir): 
    p = Path(root_dir)
    uav_folders = [x for x in p.iterdir() if x.is_dir() if not x.name.endswith("mosaic_out")]
    print(uav_folders) 

    if decision1 == '0':
        print('------------- runing desision1: LOCAL MOSAIC --------------')
        
        # create mosaic for each part
        for i in uav_folders:
            SILFILE = i.joinpath("{}_mosaicproject.xml".format('_'.join(i.name.split('_')[0:]))).as_posix()
            OUTDIR = i.parent.joinpath( "{}_mosaic_out".format('_'.join(i.name.split('_')[0:1]))).as_posix()
            MFILE = i.as_posix()
            MDFILE = i.joinpath("{}_mosaicdef.xml".format('_'.join(i.name.split('_')[0:]))).as_posix()
            
            if not os.path.isfile(SILFILE.replace("xml", "mos")):
                run_mosaic(mfile=MFILE,
                               silfile=SILFILE,
                               mdfile=MDFILE,
                               outdir=OUTDIR,
                               nodatval=NODATAVAL,
                               dbic=CHANNELS,
                               mapunits=MAPUNITS,
                               balspec="NEIGHBORHOOD", #"BUNDLE" / "NONE",
                               blend=BLEND,
                               cutmthd=CUTMTHD,
                               pxszout=PXZOUT,
                               ftype='tif',
                               sortmthd="MAXINTERSECT"
                               )
            else:
                mout_dir = Path(OUTDIR)

    mosaic_dir = [x.as_posix() for x in p.iterdir() if x.is_dir() if x.name.endswith("mosaic_out")]      
    mout_dir = Path(mosaic_dir[0])
        

    if decision2 == '0':
        print('------------- runing desision2: HOTSPOT --------------')
        # parent mosaic
        for i in mout_dir.iterdir():
            if (i.is_file()) and (i.name.endswith(".tif")):
                # run hotspot for each part
                #run_hotspot(os.path.join(mosaic_dir, i), dbic=CHANNELS, filo=os.path.join(mosaic_dir, i).split('.')[0] + '_hotspot' + '.pix')
                run_hotspot(i.as_posix(), dbic=CHANNELS)


    if decision3 == '0':
        print('------------- runing desision3: GLOBAL MOSAIC --------------')
        # mosaic all parts of submosaic					
        MFILE = mout_dir
        SILFILE = mout_dir.joinpath("mosaicproject.xml")
        MDFILE = mout_dir.joinpath("mosaicdef.xml")
        OUTDIR = mout_dir.joinpath("mosaic_out")
                        
        try:
            OUTDIR.mkdir(parents=True, exist_ok=False)	
        except FileExistsError:
            print("Folder is already there")
        else:
            print("Folder was created")   
                

        run_mosaic(mfile=MFILE.as_posix(),
                   silfile=SILFILE.as_posix(),
                   mdfile=MDFILE.as_posix(),
                   outdir=OUTDIR.as_posix(),
                   nodatval=NODATAVAL,
                   dbic=CHANNELS,
                   mapunits=MAPUNITS,
                   balspec="NEIGHBORHOOD", 
                   blend=BLEND,
                   pxszout=PXZOUT,
                   cutmthd=CUTMTHD,
                   ftype=TIF_PATTERN,
                   sortmthd="NEARESTCENTER"
                   )                            



# root_dir is the directory with images of foloowing structure
# i.e. RGB_03_32632/63-00/20200611/..tif images or CIR_05_32633/63-00/20200611/

# i.e CIR_multi_32632/DEMM_63-00/20210510_06
#     CIR_multi_32632/DEMM_63-00/20210724_06

for x in Path(root_dir).iterdir():
    if x.is_dir():
        run_process(x.as_posix())


