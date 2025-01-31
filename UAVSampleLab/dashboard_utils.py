"""
collection of dictionaries and functions for Dashboard

"""
import pandas as pd
import numpy as np


# sort after Feature Groups
sorter = {
    # Channels
      'B1': [1, 'UAV']
    , 'B2': [1, 'UAV']
    , 'B3': [1, 'UAV']
    , 'B4': [1, 'UAV']
    , 'B5': [1, 'UAV']
    , 'DSM': [1, 'UAV']

    # soil
    , 'NDSI': [2, 'soil']
    , 'OSAVI': [2, 'soil']
    , 'SAVI': [2, 'soil']

    # Chlorophyll
    , 'EVI2': [3, 'chlorophyll']
    , 'EVI': [3, 'chlorophyll']
    , 'GNDVI': [3, 'chlorophyll']
    , 'NDVI': [3, 'chlorophyll']
    , 'NDRE': [3, 'chlorophyll']
    , 'TCARI': [3, 'chlorophyll']
    , 'TCARI/OSAVI': [3, 'chlorophyll']
    , 'VDVI': [3, 'chlorophyll']

    # Plant Water
    , 'GRVI': [4, 'plant water']
    , 'GRVI2': [4, 'plant water']
    , 'NDWI': [4, 'plant water']
    , 'RDVI': [4, 'plant water']

    # Vegetation from Soil
    , 'ExG': [5, 'vegetation & soil']
    , 'ExR': [5, 'vegetation & soil']
    , 'ExGR': [5, 'vegetation & soil']
    , 'CIVE': [5, 'vegetation & soil']
    , 'MExG': [5, 'vegetation & soil']
    , 'NGRDI': [5, 'vegetation & soil']

    # another
    , 'BG': [6, 'vegetation']

    # Color Spaces
    , 'L': [7, 'color space']
    , 'A': [7, 'color space']
    , 'B': [7, 'color space']
    , 'U': [7, 'color space']
    , 'V': [7, 'color space']
    , 'H': [7, 'color space']
    , 'S': [7, 'color space']
    , 'I': [7, 'color space']
    , 'Y': [7, 'color space']
    , 'Cb': [7, 'color space']
    , 'Cr': [7, 'color space']

    , 'GLCM': [8, 'texture ']
    , 'GLDV': [8, 'texture ']

    , 'Asymmetry': [9, 'texture']
    , 'Density': [9, 'texture']
    , 'Radius of smallest enclosing ellipse': [9, 'texture']
    , 'Roundness': [9, 'texture']
}


parameter_dict = {}

for param_name in ['chlorophyll', 'soil', 'plant water','vegetation & soil', 'vegetation']:
    param = []
    for k, v in sorter.items():
        if v[1] == param_name:
            param.append(k)
        parameter_dict[param_name] = param



def feature_layer_split(df):

    #---- split to 2 dataframes 1 - Indices & Space Colors; 2 - GLCM
    searchfor = '^GLCM|^Asymm|^Elliptic|^GLDV|^Radius|^Densi|^Round'
    df1 = df[df['feature'].str.contains(searchfor, regex=True)]
    df2 = df[~df['feature'].isin(df1['feature'])]

    #print(df1.head())
    #print(df2.head())

    if not df1.empty:
        #pd.set_option('display.max_rows', None)
        #---- select only Texture features calculated for all layers
        df1['layers'] = ' Mean of all Layers'
        #df1['features'] = df1['feature']
        #print(df1['feature'].str.rsplit(' (', 1, expand=True))
        try:
            df1[['features', 'degree']] = df1['feature'].str.rsplit(' (', n=1, expand=True)
            df1['degree'] = df1['degree'].str.strip(')')
            #df1['degree'] = df1['degree'].fillna('mean of all layers')
        except:
            df1['degree'] = None
            df1['features'] = df1['feature']

        #print(df1['features'])

        for k, values in sorter.items():
            df1.loc[df1['features'].str.contains(k), 'index'] = values[0]
            df1.loc[df1['features'].str.contains(k), 'feature_level'] = values[1]

    if not df2.empty:
        #---- select only Indices & Space Colors
        #print(df2['feature'])
        df2['feature'] = df2['feature'].str.replace('(', ' ').str.strip(')')

        #st.write(df2['feature'].str.rsplit(' ', n=1, expand=True))

        df2[['features', 'layers']] = df2['feature'].str.rsplit(' ', n=1, expand=True)
        df2['layers'] = df2['layers'].str.strip(')')

        for k, values in sorter.items():
            df2.loc[df2['layers'] == k, 'index'] = values[0]
            df2.loc[df2['layers'] == k, 'feature_level'] = values[1]

        df2.sort_values(by=['index', 'feature_level', 'layers'], inplace=True)

    try:
        df1t = df1.rename(columns={'layers': 'layers2'})
        df1t = df1t.rename(columns={'degree': 'layers'}).drop(['layers2'], axis=1)
        df_main =pd.concat([df1t, df2])
        print(df_main)
    except:
        df_main = pd.concat([df1, df2])

    return {
            'df_main': df2,
            'df_texture': df1,
            'df': df_main
            }



def feature_layer_split2(df, feature_set='texture'):

    layer_dict = {}

    if feature_set == 'texture':
        #---- split to 2 dataframes 1 - Indices & Space Colors; 2 - GLCM
        searchfor_texture = '^GLCM|^Asymm|^Elliptic|^GLDV|^Radius|^Densi|^Round'
        # only texture
        df_texture = df[df['feature'].str.contains(searchfor_texture, regex=True)]

        # texture df
        if not df_texture.empty:
            dff = df_texture
            # pd.set_option('display.max_rows', None)
            # ---- select only Texture features calculated for all layers
            dff['layers'] = ' Mean of all Layers'
            # df1['features'] = df1['feature']
            # print(df1['feature'].str.rsplit(' (', 1, expand=True))
            try:
                dff[['features', 'degree']] = dff['feature'].str.rsplit(' (', n=1, expand=True)
                dff['degree'] = dff['degree'].str.strip(')')
                # df1['degree'] = df1['degree'].fillna('mean of all layers')
            except:
                dff['degree'] = None
                dff['features'] = dff['feature']

            print(dff['features'])

            for k, values in sorter.items():
                dff.loc[dff['features'].str.contains(k), 'index'] = values[0]
                dff.loc[dff['features'].str.contains(k), 'feature_level'] = values[1]

            layer_dict['df_texture'] = dff
    # without_texture
    #df_no_texture = df[~df['feature'].isin(df_texture['feature'])]

    elif feature_set == 'indices':
        searchfor_index = 'EVI|GNDV|ND|TC|VD|NDS|SAV|GRV|RDV|CI|NGR|BG|Ex'
        # only indices
        df_index = df[df['feature'].str.contains(searchfor_index, regex=True)]

        # only indices
        if not df_index.empty:
            dff = df_index
            # ---- select only Indices
            dff[['features', 'layers']] = dff['feature'].str.rsplit(' ', n=1, expand=True)

            for k, values in sorter.items():
                dff.loc[dff['layers'] == k, 'index'] = values[0]
                dff.loc[dff['layers'] == k, 'feature_level'] = values[1]

            dff.sort_values(by=['index', 'feature_level', 'layers'], inplace=True)
            layer_dict['df_indices'] = dff

    elif feature_set == 'color spaces':

        #---- split to 2 dataframes 1 - Indices & Space Colors; 2 - GLCM
        searchfor_texture = '^GLCM|^Asymm|^Elliptic|^GLDV|^Radius|^Densi|^Round'
        # only texture
        df_texture = df[df['feature'].str.contains(searchfor_texture, regex=True)]

        searchfor_index = 'EVI|GNDV|ND|TC|VD|NDS|SAV|GRV|RDV|CI|NGR|BG|Ex'
        # only indices
        df_index = df[df['feature'].str.contains(searchfor_index, regex=True)]
        df_no_index = df[~df['feature'].isin(df_index['feature'])]
        df_space = df_no_index[~df_no_index['feature'].isin(df_texture['feature'])]

        # only color_spaces
        if not df_space.empty:
            dff = df_space
            #---- select only Indices

            #df['feature'] = df['feature'].str.replace('(', ' ').str.strip(')')

            dff[['features', 'layers']] = dff['feature'].str.rsplit(' ', n=1, expand=True)
           # df['layers'] = df['layers'].str.strip(')')

            print(dff['feature'])

            for k, values in sorter.items():
                dff.loc[dff['layers'] == k, 'index'] = values[0]
                dff.loc[dff['layers'] == k, 'feature_level'] = values[1]
            dff.sort_values(by=['index', 'feature_level', 'layers'], inplace=True)
            print(dff['feature'])
            layer_dict['df_col_spaces'] = dff

    df_index, dff, df_space, df_no_index, df_no_texture, df1t, df_main, df_texture = None, None, None, None, None, None, None, None

    return layer_dict


def generate_dict_from_dict_values(dictionary, key=None, val=1):
    '''
    generates dict from a values of dict
    dictionary: dict
       dictionary with  list of values
    key: int
        position of the value to be set to key of dictionary
    val: int
        position of the value to be set to value of dictionary
     '''
    res = dict()
    for k, sub in dictionary.items():
        if key == None:
            res[k] = sub[val]
        else:
            if isinstance(key, int):
                key = key
                res[sub[key]] = sub[val]
    return res



def resample_values(data,
                    group_by=['aoi', 'year', 'sl_nr', 'variable', 'crop_type_code'],
                    val_column='value',
                    column_to_fill=['aoi', 'year', 'sl_nr','crop_type_code','variable']):
    '''
    data: pandas dataframe
        datframe of long format with index as date
        use this to convert date to datetime:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)

    group_by: list of str
        list of columns the dataframe to be grouped by
    val_column: str
        column name with values to be resampled
    column_to_fill: list of str
        columns with values to be copied/filled

    Returns:
    ---------
        pandas dataframe with resampled values
    '''

    df_list=[]
    for i, d in data.groupby(group_by):
        # convert value column to float
        d[val_column] = d[val_column].astype(float)

        # Create a date range with desired frequency (e.g., daily)
        start_date = d.index.min()
        end_date = d.index.max()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')


        # Find duplicate dates in the index
        duplicates = d.index[d.index.duplicated(keep=False)]

        # handle duplicate values
        if len(duplicates) > 0:
            # Create a DataFrame subset containing the duplicate rows
            df_duplicates = d[d.index.isin(duplicates)]

            # Print the DataFrame subset
            print("DataFrame subset containing duplicate dates:")
            print(df_duplicates)
            d = d.loc[~d.index.duplicated(keep='first')]
            # Calculate the mean for the duplicate rows
            #TODO: implement mean of duplicate values!!!!!
            #df_mean = d.groupby(d.index).mean()
        else:
            print("No duplicates found in the DataFrame index.")



        # Reindex the DataFrame with the date range and fill missing values with NaN
        dt = d.reindex(date_range)

        # add new column  'origin_bbch' to df that indicates whether bbch_code is measured or was interpolated
        column_of_origin = '{}_origin'.format(val_column)
        dt[column_of_origin] = np.where(dt[val_column].notnull(), 'measured', 'interpolated')


        # Interpolate values for missing dates
        dt[val_column] = dt[val_column].interpolate()

        #st.write(dt)


        # fill interpolated 'NAN' values in columns with existing values above
        for col in column_to_fill:
            dt[col] = dt[col].fillna(method='ffill')

        #dt[column_of_origin] = np.where(dt[val_column].notnull(), 'interpolated', np.nan)
        # collect subdataframes in a list
        df_list.append(dt.reset_index(drop=False).rename(columns={'index': 'date'}))

    #concat all subdatframes into one
    dfr = pd.concat(df_list, ignore_index=True)

    return dfr