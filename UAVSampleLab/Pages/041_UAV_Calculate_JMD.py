'''
Script to Calculate Jeffries Matusita Distance (JMD) for Sample values and import JMD values into DB
Samples collected in eCognition Chessboard Segmentation

Source: Niklas Schmidt (https://github.com/Niklas-Schm/Bachelorarbeit)
'''
from decouple import config
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd


import config as cfg
from dbflow.src import db_utility as dbu



def Jeffries_Matusita_Distance(mean_class1, mean_class2, CovarianceMatrix_class1, CovarianceMatrix_class2):
    '''
    Function calculates the Jeffries Matusita Distance
    mean_class1: float
        mean of first class
    mean_class2: float
        mean of second class
    covariance_matrix_class1: numpy.array
        Covariance Matrix of first class
    CovarianceMatrix_class2: numpy.array
        Covariance Matrix of second class
    ------
    return: 'numpy.matrix'
    '''

    # Create variables for the formula
    m1 = np.array(mean_class1)
    s1 = np.array(CovarianceMatrix_class1)

    m2 = np.array(mean_class2)
    s2 = np.array(CovarianceMatrix_class2)
    print('m1: {}'.format(m1))
    print('m2: {}'.format(m2))
    print('s1: {}'.format(s1))
    print('s2: {}'.format(s2))

    # create first term of Bhattacharyya Distance
    # subtract means and create a 2-dimensional matrix
    m12 = np.subtract(m1, m2)
    print('m12_: {}'.format(m12))

    # transpose m12
    m12_2D = np.array([m12])
    m12t = m12_2D.T

    # add covariances and create a 2-dimensional matrix
    s12 = np.add(s1, s2)/2
    matrix_s12 = np.asmatrix(s12)
    print('s12: {}'.format(s12))

    # calculate the inverse matrix of s12
    try:
        s12i = np.linalg.inv(matrix_s12)
    except:
        #print(matrix_s12)
        #print(mean_class1, mean_class2, CovarianceMatrix_class1, CovarianceMatrix_class2)
        #sys.exit()
        return np.nan


    # combine to first Term of Bhattacharyya Distance
    B1 = (1/8) * m12 * (s12i * m12t)

    print('Bhattacharyya distance 1: {}'.format(B1.item(0)))

    #  second Term of Bhattacharyya Distance
    # convert to matrix
    s1_matrix = np.asmatrix(s1)
    s2_matrix = np.asmatrix(s2)

    # calculate determinate for s1 and s2
    ds1 = np.linalg.det(s1_matrix)
    ds2 = np.linalg.det(s2_matrix)

    print('ds1: {}'.format(ds1))
    print('ds2: {}'.format(ds2))

    # calculate determinate for s1 + s2
    ds12 = np.linalg.det(s1_matrix + s2_matrix)

    # divide ds12 by2
    ds12_half = ds12 / 2

    # calculate square root of ds1 * ds2
    sq_ds1_ds2 = np.sqrt(ds1 * ds2)

    # combine to second part of Bhattacharyya Distance
    B2 = (1/2) * np.log(ds12_half / sq_ds1_ds2)

    print('Bhattacharyya distance 2: {}'.format(B2))

    #  combine to Bhattacharyya Distance
    B = B1.item(0) + B2

    #  calculate J-M separability
    JMD = 2 * (1 - np.exp(-1 * B))

    return JMD


def jmd_wrapper(gdf, select_class_1, select_class_2):

    # Create Df for each Class
    df_class1 = gdf.loc[gdf['class_name'] == select_class_1].dropna()
    # df_class1_wide = pd.pivot(df_class1, index=None, columns=['feature'], values='value') #Reshape from long to wide
    df_class2 = gdf.loc[gdf['class_name'] == select_class_2].dropna()

    print('df_class1: {}'.format(df_class1))
    print('df_class1: {}'.format(df_class2))

    output = pd.DataFrame()
    index = []
    JMD_value = []

    Indices = gdf['feature'].unique()

    # Indices = gdf.columns[gdf.columns.str.contains('Mean')].tolist()
    # print(Indices)

    for i in Indices:
        print('INDEX: {}'.format(i))
        mean_1 = df_class1['value'].loc[df_class1['feature'] == i].mean()
        mean_2 = df_class2['value'].loc[df_class2['feature'] == i].mean()
        # mean_1 = df_class1[[i]].mean(numeric_only=True)
        # mean_2 = df_class2[[i]].mean(numeric_only=True)
        print(df_class1['value'].loc[df_class1['feature'] == i])
        print(df_class2['value'].loc[df_class2['feature'] == i])

        print('mean_1: {}'.format(mean_1))
        print('mean_2: {}'.format(mean_2))

        Cov_class1 = np.cov(df_class1['value'].loc[df_class1['feature'] == i], bias=True)
        Cov_class2 = np.cov(df_class2['value'].loc[df_class2['feature'] == i], bias=True)
        # Cov_class1 = np.cov(df_class1[[i]], bias=True, rowvar=False)
        # Cov_class2 = np.cov(df_class2[[i]], bias=True, rowvar=False)

        JMD = Jeffries_Matusita_Distance(mean_1, mean_2, Cov_class1, Cov_class2)
        print('JMD: {}'.format(JMD))
        index.append(i)

        if JMD != np.nan:
            JMD_value.append(round(JMD, 2))
        else:
            print('JMD is nan !!! ')
            JMD_value.append(JMD)

        #if JMD <= 1:
        #    # print('geringe Trennbarkeit', i, JMD)
        #    index.append(i)
        #    JMD_value.append(JMD.round(2))
        #else:
        #    # print('hohe Trennbarkeit', i, JMD)
        #    index.append(i)
        #    try:
        #        JMD_value.append(JMD.round(2))
        #    except:
        #        JMD_value.append(JMD)

    # test = Jeffries_Matusita_Distance(df_class1.loc[df_class1['feature'] == 'NDVI'].mean(numeric_only=True), df_class2.loc[df_class2['feature'] == 'NDVI'].mean(numeric_only=True), np.cov(df_class1.loc[df_class1['feature'] == 'NDVI']), np.cov(df_class2.loc[df_class2['feature'] == 'NDVI']))
    # print(test)
    output['feature'] = index
    output['jmd'] = JMD_value
    output['class1'] = select_class_1
    output['class2'] = select_class_2
    output['refpanel'] = df_class1['refpanel'].loc[df_class1['class_name'] == select_class_1].unique().tolist()[0]
    #output['refpanel2'] = df_class1['refpanel'].loc[df_class2['class_name'] == select_class_2].unique().tolist()[0]
    output['sensor1'] = df_class1['sensor'].loc[df_class1['class_name'] == select_class_1].unique().tolist()[0]
    output['sensor2'] = df_class2['sensor'].loc[df_class2['class_name'] == select_class_2].unique().tolist()[0]
    output['jmd'] = output['jmd'].astype(float)
    output = output.sort_values('jmd', ascending=False)

    return output



# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

sql_filter = f"""
                SELECT 
                      crop_type_code, 
                      value, 
                      refpanel, 
                      feature
                FROM uavfeaturestatistic
                WHERE 
                    value != "undefined"
                    AND refpanel != "noRP"
                    AND feature != "Area (Pxl)"
                    AND feature NOT LIKE "DS%"
                GROUP BY 
                crop_type_code
                ;
            """

select_df = dbu.query_sql(sql_filter, db_engine=cfg.dbarchive.archive.engine)
CROP_TYPE_CODE = select_df['crop_type_code'].unique().tolist()

#  streamlit filter
col01, col02 = st.columns((3, 4))
with col01:
    CROP_TYPE = st.selectbox(label='Select Crop Type:', options=CROP_TYPE_CODE)
with col02:
    REF_PANEL = st.selectbox(label='Select Unit:', options=['reflectance', 'DN - without Ref. Panel'])
    #AOI = st.selectbox(label='Select AOI:', options=aoi_data['aoi'].unique())




if REF_PANEL == 'reflectance':
    UNIT = "RP"
elif REF_PANEL == 'DN - without Ref. Panel':
    UNIT = "noRP"

if CROP_TYPE:
    sql_filter = f"""          
            SELECT 
                class_name, 
                feature, 
                ROUND(value, 2) as value, 
                aoi, 
                uav_date, 
                bbch, 
                crop_type_code, 
                fid,
                refpanel, 
                sensor
            FROM uavfeaturestatistic
            --WHERE feature like "Mean%"
            WHERE 
                value != "undefined"
                AND refpanel == "{UNIT}"
                AND feature != "Area (Pxl)"
                AND feature NOT LIKE "DS%"
                AND crop_type_code ==  "{CROP_TYPE}"
            --AND feature == "Compactness" 
            ;
            """

    gdf = dbu.query_sql(sql_filter, db_engine=cfg.dbarchive.archive.engine)
    gdf = gdf.apply(pd.to_numeric, errors='ignore')
    gdf['value'] = pd.to_numeric(gdf['value'], errors='coerce')


    #gdf_wide = pd.pivot(gdf, index=None, columns=['class_name', 'feature'], values='value') #Reshape from long to wide
    #print(gdf_wide)

    classes = gdf['class_name'].unique()

    col001, col002 = st.columns((4, 4))
    with col01:
        if st.button("Calculate for all Pattern Combinations & Import to DB"):

            # collect all possible combinations of the pattern pairs for JMD
            class_pairs = []
            for a in sorted(classes):
                for b in sorted(classes):
                    if not b == a:
                        if int(b.split('_')[1]) - int(a.split('_')[1]) < 2:
                            if int(a.split('_')[1]) - int(b.split('_')[1]) < 2:
                                if not [b, a] in class_pairs:
                                    select = [a, b]
                                    # print(a, b)
                                    class_pairs.append(select)
            #st.write(class_pairs)
            for a, b in class_pairs:
                select_class_1 = a
                select_class_2 = b
                jmd_result = jmd_wrapper(gdf, select_class_1, select_class_2)

                # this needed to reopen the db in each update in the same threath
                db_path_2 = config('DB_PATH_2')
                db = dbu.connect2db(db_path_2)
                # ------- Insert Results
                pim_key = db.get_primary_keys('uavfeaturejmd')
                db.insert(table='uavfeaturejmd', primary_key=pim_key,
                        orderly_data=jmd_result.to_dict(orient='records'), verbose=True, update=True)
                print('Inserted for classes: {} - {}'.format(select_class_1, select_class_1))
            print('Finished Inserting Data to DB !')


    st.markdown('')
    st.markdown('OR select for each Pattern Combination:')
    st.markdown('')

    col01, col02 = st.columns((4, 4))
    with col01:
        select_class_1 = st.selectbox(label='Select class_1:', options=sorted(classes))
    with col02:
        select_class_2 = st.selectbox(label='Select class_2:', options=sorted(classes))

    jmd_result = jmd_wrapper(gdf, select_class_1, select_class_2)

    ### ploting
    JMD_barchart = alt.Chart(jmd_result).mark_square(size=60).encode(
        alt.X('jmd'),
        alt.Y('feature', sort='-x', axis=alt.Axis(grid=True)),
        # alt.Y('Indices'),
        tooltip='jmd:Q',
    )
    line = alt.Chart(pd.DataFrame({'jmd': [1.3]})).mark_rule(color='red', size=2).encode(
        x='jmd')  # 1.3 as JMD Value Threshold

    #### Layout
    col01, col02 = st.columns(2)
    with col01:
        st.write(jmd_result)
    with col02:
        st.write(jmd_result)
    with st.expander('show plot of JMD(s):'):
        st.write(JMD_barchart + line)

    filename = f"{select_class_1}_{select_class_2}.csv"
    output_download = jmd_result.to_csv(sep=';', index=False).encode('utf-8')

    #### Layout
    col11, col22 = st.columns(2)
    with col11:

        st.download_button(
            label="Download data as CSV",
            data=output_download,
            file_name=filename,
            mime='csv',
        )
        st.write(filename)



    with col22:
        if st.button("Import to DB"):
            data_load_state = st.text('Loading data...')
            # this needed to reopen the db in each update in the same threath
            db_path_2 = config('DB_PATH_2')
            db = dbu.connect2db(db_path_2)
            # ------- Insert Results
            pim_key = db.get_primary_keys('uavfeaturejmd')
            db.insert(table='uavfeaturejmd', primary_key=pim_key,
                      orderly_data=jmd_result.to_dict(orient='records'), verbose=True, update=True)

            data_load_state.text("Done!")

#dbu.dbarchive.archive.close()