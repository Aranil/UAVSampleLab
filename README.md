# UAVSampleLab
A collection of scripts designed to analyze UAV data and assess sample complexity, 
specifically for use in Convolutional Neural Network (CNN) classification tasks.


This repository contains the scripts used in the analysis presented in the paper:
**[Assessing Data and Sample Complexity in Unmanned Aerial Vehicle Imagery for Agricultural Pattern Classification](https://www.sciencedirect.com/science/article/pii/S2772375525000334?via%3Dihub)**


*****for Section 4.2.1. Selected Features and Layers***** 

 UAVSampleLab/Pages/041_UAV_Calculate_JMD.py
 UAVSampleLab/Pages/042_UAV_Feature_Analysis_heat_map.py  
 UAVSampleLab/Pages/045_UAV_Feature_Importance_plot.py


*****for Section 4.2.2. Class Variability***** 

UAVSampleLab/texture_analysis/semivariogram.py 





## 📁 Project Structure

```plaintext
UAVSampleLab/

└── tests/
    ├── data/                   # contains data used to test functions
└── _DB/
    ├── statistic_eCognition    # containse .txt files with accuracy assesment exported from eCognition
    ├── RCM_main_analysis_uav.db 
└── _logs/
└── _temp/                      # to store intermediate calcualtions
│ 
└── UAVSampleLab/
    ├── accuracy_assessment/     # of CNN model perfomance
    │   ├── accuracy_utils.py                 # function definitions to calulate accuracy metrics 
    │   ├── import_accuracy_assesment2db.py   # Step 1 to import accuracy assesment in .txt file (collected fid-wise in eCognition) to DB 
    │   ├── _plot_accuracy_metrics.py            
    │   └── _plot_confusion_matrix.py            
    │    
    ├── create_CNN_samples/      # all files in this part use geoutils.raster.py 
    │   ├── crop_uav_images.py          # Step 1. to crop UAV image after WKT (field border with inner buffereing) retrived from RCM.db -> to get rid off close to border areas, to resample and calulate Color Spaces, LBP, GLCM
    │   ├── crop_uav_samples.py         # Step 2. to crop sample patches from UAV image with vector file exported from eCognition
    │   ├── mosaic_sample_patches.py    # Step 3. to mosaic all sample patches to sample patch matrix
    │   └── xml_utils.py                # used in mosaic_sample_patches.py
    │ 
    ├── Lab.py                   # Main entry point for the Streamlit app
    ├── pages/                   # Additional Streamlit pages (auto-discovered)
    │   ├── 041_UAV_Calculate_JMD.py              # Jeffries-Matusita Distance
    │   ├── 042_UAV_Feature_Analysis_heat_map.py  # Heat map visualization
    │   ├── 045_UAV_Feature_Importance.py         # Generate Feature Importance
    │   ├── 045_UAV_Feature_Importance_plot.py    # Feature importance analysis
    │   ├── 047_UAV_membership_function.py        # Visualisation of membership function results for pre-selected Input Layers (exported results from eCognition)
    │   └── 049_UAV_CNNLayerStatistic_UMAP.py     # Representation of the spectral information of selectd Input Layers for CNN models
    │ 
    ├── texture_analysis/        # Backend analysis utilities
    │   └── semivariogram.py                      # Texture information analysis
    │ 
    ├── uav_processing/          # additional scripts apply before analysis 
    │        ├── catalyst_mosaic.py                # Step 1: script for color balancing of image subset after UAV image preprocessing in Metashape Agisoft (run outside this module)
    │        ├── convert_64bit_to_32_bit.py        # Step 2: additional script, ran if required after UAV image preprocessing (used in geoutils.raster.py)
    │        └── cut_header_name.py                          
    │ 
    ├── custom/                                             # for imported dbflow module
    │        └── sql/  
    │          └── create_table/  # contains the sql scripts to create table required for the DB
    │                   ├──  _create_uavcnnaccuracyreport.txt  
    │                   ├──  ...       
    ├── geoutils/
    │        ├── __init__.py    
    │        └── raster.py      
    │     
    ├── colors.py 
    ├── config.ini  
    ├── dashboard_utils.py   
    ├── .env   
└── environment.yml              
    
```

### ▶️ Run the App
Open your terminal or command prompt and follow these steps:

```bash
# 1. Activate your conda environment
conda activate uav_dashboard

#2. Navigate to the directory containing Lab.py
cd path/to/UAVSampleLab/UAVSampleLab

# 3. Run the Streamlit app
streamlit run UAVSampleLab\Lab.py [ARGUMENTS]
```

### ▶️ UAV Image Processing Steps

![UAV image processing workflow for analysis ready data.](https://ars.els-cdn.com/content/image/1-s2.0-S2772375525000334-gr003_lrg.jpg)

1. Agisoft_Batch_RGB(CIR).xml  - XML Workflow for Image processing in Metshape (semimanual, requires the control of intermediate results, for parameter detail see manual https://www.agisoft.com/pdf/metashape-pro_1_5_en.pdf)
2. catalyst_mosaic.py - Python Script  for Image Enhancement and Mosaicking (automated)
3. convert_64bit_to_32_bit.py, cut_header_name.py, crop_uav_samples.py - additional scripts



Contributors: [Niklas Schmitt](https://github.com/Niklas-Schm/), 
              [Friederike Metz](https://github.com/friedmag-m),
              Sören Hese
