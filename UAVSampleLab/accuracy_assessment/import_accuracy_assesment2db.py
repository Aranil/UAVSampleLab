'''
Reads exported from eCognition csv files with calculated confusion metrics (since the accuracy assessment in the
Software was calculated for one image/fid_date at a time)
(Re-)Calculates the accuracy metrics for all images (fid_date) and sub-classes
Imports the Results in DB table -> 'uavcnnaccuracy' and 'uavcnnaccuracyreport'
'''
#requires pip install disarray and import disarray
#import disarray

import pathlib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from dbflow.src import db_utility as dbu
from accuracy_assessment import accuracy_utils as acc



def read_eCognition_statistics(csv_root, pattern=None):
    '''
    Reads the accuracy metrics exported from eCognition (one .csv file for each assessed UAV image), as well as the
    parameters defined in file header - 'WW_CIR_FRIEN_82-00_20210510_xx_reflectance_no.csv'

    WW -> crop_type_code (KM, SG, WR)
    CIR -> or  'RGB' -sensor time
    DEMM -> study area
    231-01 -> fid
    20210510 -> date of image acquisition
    xx -> placeholder for pattern
    reflectance -> or 'DN' image unit
    _no -> 'no' are samples not used for model training, 'yes' are samples used for model training

    Parameters:
        csv_root: str
            path to the folder with csv files
        pattern: str
            'xx' or 'method9' -> an additional label to label the different methods of classification or models to be
             able to select different methods from db

    Returns:
        df
    '''

    ifile = pathlib.Path(csv_root)

    for j, i in enumerate(ifile.iterdir()):

        print(i.name.split('_'))

        if len(i.stem.split('_')) == 8:
            crop_type_code, sensor, aoi, sl_nr, date, pattern_, unit, usedInCNN = i.stem.split('_')
        elif len(i.stem.split('_')) == 9:
            crop_type_code, sensor, aoi, sl_nr, date, _, unit, usedInCNN, pattern_ = i.stem.split('_')

        if pattern != None:
            pattern_ = pattern


        df = pd.read_csv(i.as_posix(), sep=';', decimal=' ', encoding='ANSI', engine='python', dtype='str',
                         skipinitialspace=True)

        df_collect = pd.DataFrame()

        # extract matrix from df => header definition according to eCognition output file
        df_subset = df.loc[df['User Class \ Sample'].isin(cfg.classes)].iloc[:, :-2]

        for k in df_subset['User Class \ Sample'].values.tolist():
            df = pd.melt(df_subset, id_vars=['User Class \ Sample'],
                         value_vars=[k], value_name='user_class').rename(
                columns={"User Class \ Sample": "label_predicted", "variable": "label_actual"}, )

            df_collect = pd.concat([df_collect, df], ignore_index=True)

        df_collect['id'] = j
        df_collect['date'] = date
        df_collect['sl_nr'] = sl_nr
        df_collect['crop_type_code'] = crop_type_code
        df_collect['pattern'] = pattern_
        df_collect['unit'] = unit
        df_collect['sample_used_in_cnn'] = usedInCNN

    return df_collect

def plot_cmatrix(dfA, title=None):
    dfA.columns = [acc.tick_labels.get(col, col) for col in dfA.columns]  # Replace class names with tick labels
    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(dfA, annot=True, fmt='d', cmap='Blues', cbar=False, square=True, linewidths=0.5)

    # Add labels and title
    plt.xlabel("Predicted", labelpad=20)
    plt.ylabel("True", labelpad=20)
    plt.title(title)
    plt.show()




if __name__ == '__main__':

    dbarchive = dbu.connect2db(cfg.db_path)

    #----------------------------------------------------------------------------------------------
    #--------- Folder with csv files exported from eCognition - accuracy assement for each field !

    idata_path = pathlib.Path(cfg.db_path)

    #csv_root = idata_path.joinpath(r'_idata\accuracy_metrics', 'accuracy_WW_CIR')
    csv_root = idata_path.parent.joinpath(r'Statistic_eCognition\accuracy_metrics', 'accuracy_WW_CIR_B5_BG_CIVE_EVI2') # accuracy of the model trained for 7 classes
    #csv_root = r'...\_Accuracy\accuracy_WW_CIR_B5_CIVE_EVI2_GRVI_NDSI' # accuracy of the model trained for 6 classes

    # ----------------------------------------------------------------------------------------------


    import2db_accuracy_matrix = True
    insert2db = False
    pattern = 'xx' # 'xx', 'method9', etc

    if import2db_accuracy_matrix == True:

        df = read_eCognition_statistics(csv_root, pattern=None)
        print(df.head())

        if insert2db == True:
            # insert the  data to a DB
            dbarchive.insert(table='uavcnnaccuracy',
                                 primary_key=dbarchive.get_primary_keys('uavcnnaccuracy'),
                                 orderly_data=df.to_dict(orient='records'),
                                 update=True)





    calulate_accuracy_metrics = True
    insert2db = False
    normalize_confusion_matrix = True # select if normalized or non normalized values should be inserted into db
    plot_accuracy_matrix = True


    if calulate_accuracy_metrics:

        sql = f"""
                SELECT 
                
                label_actual, 
                label_predicted, 
                sum(user_class) as user_class, 
                sl_nr, 
                crop_type_code, 
                pattern, 
                unit, 
                sample_used_in_cnn
                
                FROM 
                    uavcnnaccuracy
                WHERE
                	pattern=='{pattern}'
                --	user_class != 0
                --AND unit== 'reflectance'
                --AND unit== 'DN'
                --AND label_actual=='bare_soil'
                --AND label_predicted=='bare_soil'
                --AND 
                --pattern NOT LIKE 'mosmethod%'
                
                --AND unit == 'reflectance'
                --AND sample_used_in_cnn == 'no'
                --AND crop_type_code == 'WR'
                --AND pattern == 'Blure'
                
               -- WHERE 
               -- pattern LIKE 'mosmethod%'
                --sl_nr == '190-00'
                
                GROUP BY 
                    label_actual, 
                    label_predicted,
                    --sl_nr, 
                    crop_type_code, 
                    pattern, 
                    unit, 
                    sample_used_in_cnn
                
                ORDER BY
                    unit,
                    pattern,
                    label_actual,
                    label_predicted,
                    pattern
                ;
    
        """
        df_query = pd.read_sql(sql, dbarchive.archive.engine)

        #-------- 1. Prepare data to Calculate Confusion Matrix
        # group by the data to be plotted after
        for j, (i, df) in enumerate(df_query.groupby(['crop_type_code', 'pattern', 'unit', 'sample_used_in_cnn'])):

            crop_type_code = df['crop_type_code'].unique()
            pattern = df['pattern'].unique()
            used_in_training = df['sample_used_in_cnn'].unique()


            dff = df.groupby(['crop_type_code', 'pattern', 'unit', 'sample_used_in_cnn', 'label_actual', 'label_predicted'])['user_class'].sum().reset_index()
            df_main = dff[['label_actual', 'label_predicted', 'user_class']]


            # select labels with non zero values at user_class and actual class
            non_zero_labels = df_main.query("user_class!=0")
            labels_ = list(pd.unique(non_zero_labels[['label_actual', 'label_predicted']].values.ravel()))
            selected_labels = df_main.query("label_actual == @labels_ & label_predicted == @labels_" )

            # convert back to origin form
            dfr = selected_labels.pivot(index='label_predicted', values='user_class', columns='label_actual')
            print('+'*20)


            # this is confusion matrix for all classes
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                #print(f.reset_index().iloc[:, 1:]) # if need to exclude micro-average
                print(dfr)

            # -------- 2. Calculate Confusion Matrix
            dfr = dfr.replace(np.nan, 0)
            class_names = dfr.columns
            A = dfr.reset_index().iloc[:, 1:].astype(int).to_numpy()
            print('confusion matrix for all classes together:')
            print(A)


            if normalize_confusion_matrix == True:

                # normalize the confusion matrix
                normalized_A = A.astype('float') / A.sum(axis=1)[:, np.newaxis]

                print('Normalized confusion matrix for all classes together:')
                print(normalized_A)

                # convert nan to 0, round value, convert to int
                A = np.round(np.nan_to_num(normalized_A * 100), 0).astype(int)
                normalized_A = None



            # ----  calculate number of True Positive (TP) & True Negative (TN) samples
            TP, FP, FN, TN, TotalSamp = acc.classification_report(A, classes=class_names)
            print('-' * 80)

            # Instantiate the confusion matrix DataFrame with index and columns
            # dtype=int is important for Windows users
            dfA = pd.DataFrame(A, index=class_names, columns=class_names, dtype=int)

            # plotting here is only for monitoring of the result before inserting to db!
            if plot_accuracy_matrix == True:
                plot_cmatrix(dfA, title=f"Confusion Matrix Heatmap for {crop_type_code}_{used_in_training}")


            UA = TP / (TP + FN) #* 100 # User Accuracy (UA)
            PA = TP / (TP + FP) #* 100 # Producer Accuracy (PA)
            OE = FN / (FN + TN) #* 100 #Omission Error (OE)

            # https://www.johndcook.com/blog/2018/09/06/accuracy-precision-and-recall/
            df_report = acc.confusion_metrics(dfA)

            df_collect = pd.DataFrame()
            df_long = pd.melt(df_report.reset_index(), id_vars='class', var_name='metric', value_name='value')
            df_collect = pd.concat([df_collect, df_long], ignore_index=True)

            # add total_sum_TP & total_sum_TN to the df with metrics
            # total_sum_TP
            class_report_dict = { 'TP': TP
                                , 'FP': FP
                                , 'TN': TN
                                , 'FN': FN
                                , 'Total':  TotalSamp
                                , 'OE': OE
                                , 'user_accuracy': UA
                                , 'producer_accuracy':  PA
                                }

            # -------- 3. Collect all Metrics into df for DB import
            for k, v in class_report_dict.items():
                for _class, val in zip(class_names.tolist(), v):
                    new_row = {
                        #'id': None
                        # 'date': None
                          'metric': 'number_of_{}_samples'.format(k)
                        , 'class': _class
                        , 'value': val
                        #, 'sl_nr': None
                        #, 'crop_type_code': None
                        , 'pattern': None
                        #, 'unit': None
                        #, 'sample_used_in_cnn': None
                    }
                    df_collect = pd.concat([df_collect, pd.DataFrame([new_row])], ignore_index=True)

            # add new columns
            #df_collect['sl_nr'] = df['sl_nr'].unique()[0]
            df_collect['id'] = j
            df_collect['date'] = None
            df_collect['sl_nr'] = None
            df_collect['crop_type_code'] = dff['crop_type_code'].unique()[0]
            df_collect['pattern'] = dff['pattern'].unique()[0]
            df_collect['unit'] = dff['unit'].unique()[0]
            df_collect['sample_used_in_cnn'] = dff['sample_used_in_cnn'].unique()[0]


            # -------- 4. Insert to DB
            if insert2db == True:
                # insert the  data to a DB
                dbarchive.insert(table='uavcnnaccuracyreport',
                                  primary_key=dbarchive.get_primary_keys('uavcnnaccuracyreport'),
                                  orderly_data=df_collect.to_dict(orient='records'),
                                  update=True)
