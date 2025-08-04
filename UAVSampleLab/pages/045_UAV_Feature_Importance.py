"""
Script to Calculate and Visualise feature Importance according to different Classifiers
The Results can be imported to DB
"""

import streamlit as st
import numpy as np
import pandas as pd
import pathlib

# Supervised learning classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance

# unsupervised learning classifiers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import sklearn

import dashboard_utils as dutils
import config as cfg
from dbflow.src import db_utility as dbu




st.write('sklearn version: {}'.format(sklearn.__version__))



col01, col02, col03, col04, col05, col06 = st.columns((2, 3, 2, 3, 3, 2))
with col01:
    CROP_TYPE = st.selectbox(label='Select Crop Type:', options=['SG', 'WG', 'WW', 'WR', 'ZwFr', 'KM', 'SM'])
with col02:
    JMD = st.slider(label='JMD threshold:', min_value=1.0, max_value=2.0, value=1.9, step=0.1)
with col03:
    BBCH_GROUP = st.selectbox(label='Select BBCH group:', options=[1, 2, 3, 4, 5, 6, 7, 8, 9])
with col04:
    TOP_FEATURES = st.slider(label='Top Features :', min_value=1, max_value=150, value=10, step=1)
with col05:
    SAMPLE_LIMIT = st.slider(label='Sample Limit :', min_value=300, max_value=1000, value=1000, step=100)


# Step 3: Read the unique values back from the text file
with open(r'{}\_temp\unique_features_{}.txt'.format(pathlib.Path(cfg.output_path).parent, CROP_TYPE), 'r') as f:
    read_features = f.read().splitlines()


# Convert the read values to a list (if needed)
read_features_list = list(read_features)


sql_ = f"""
        SELECT feature
             FROM (
                        SELECT
                            feature, 
                            jmd, 
                            substr(class1, 4, instr(class1, '_') - 2) as bbch_group,
                            --class1, 
                            --class2,
                            substr(class1, 0, instr(class1, '_')) as crop,

                            COUNT(jmd) as count_jmd

                        FROM
                            uavfeaturejmd

                        WHERE 
                            jmd >= {JMD}

                        AND 
                        class1  LIKE '{CROP_TYPE}%'

                        AND bbch_group == '{str(BBCH_GROUP)}'
                        --AND bbch_group IN ('1', '2', '3', '4', '5', '6', '7', '8', '9')
                        
                        GROUP BY
                            feature
                            --crop
                            --class1, 
                            --class2

                        ORDER BY 
                            --crop,
                            COUNT(jmd) DESC,
                            feature
                            ) 
            WHERE feature in ({str(read_features_list)[1:-1]})
            --WHERE count_jmd >= 25
            LIMIT {str(TOP_FEATURES)}
        ;
"""
# AND bbch_group == '{str(BBCH_GROUP)}'

dt_hist = dbu.query_sql(sql_, db_engine=cfg.dbarchive.archive.engine)
data_hist = dutils.feature_layer_split(dt_hist)['df']

FEATURES = data_hist['feature'].unique().tolist()

data_hist = None


R_Panel = st.selectbox(label='Reference Panel:', options=['RP', 'noRP', 'all'])

if (R_Panel == 'RP') or (R_Panel == 'noRP'):
    sql_text = f"""

                AND refpanel == '{R_Panel}'

                """
else:
    sql_text = f"""

                """

#to select certian bbch
bbch_sql = f"""  AND
                 bbch_group == '{BBCH_GROUP}'  
            """



# -----------------------------------------------
#  plot Feature Importance
# -----------------------------------------------
sql_filter = f"""SELECT
                --inner_x, 
                --inner_y,
                aoi,
                uav_date,
                bbch,
                sensor,
                value,
                feature,
                fid,
                refpanel,
                class_name,
                substr(class_name, 0, instr(class_name, '_')) as crop,
                substr(class_name, 4, instr(class_name, '_') - 2) as bbch_group,
                substr(class_name, 6, instr(class_name, '_') - 2) as damage,
                substr(class_name, 8, instr(class_name, '_') - 2) as pattern,
                substr(class_name, 6, instr(class_name, '_')+2) as damage_pattern
                FROM
                uavfeaturestatistic
                WHERE

                class_name LIKE '{CROP_TYPE}%'
                {sql_text}

                AND
                feature IN ({str(FEATURES)[1:-1]})

                AND (feature NOT LIKE '%DSM' OR feature NOT LIKE '%DSM%')
                AND feature NOT LIKE 'GLCM%' AND feature NOT LIKE 'GLDV%' 
                AND feature NOT LIKE 'Round%' AND feature NOT LIKE 'Radius%' 
                AND feature NOT LIKE 'mode%'
                AND feature NOT LIKE 'Dens%' AND feature NOT LIKE 'Ellip%' 

                --GROUP
                --BY
                --inner_x, 
	            --inner_y,
                --feature,
                --crop,
                --bbch_group,
                --damage,
                --pattern
                ;
"""
sql_dt = dbu.query_sql(sql_filter, db_engine=cfg.dbarchive.archive.engine)
data = dutils.feature_layer_split(sql_dt)['df'] #-> for all features, ['df'] -> for features except texture

# convert some values to categorical data
data['bbch'] = data['bbch'].astype('category')
data['uav_date'] = data['uav_date'].astype('category')
#data['inner_x'] = data['inner_x'].astype('category')
#data['inner_y'] = data['inner_y'].astype('category')
data['value'] = pd.to_numeric(data['value'], errors='coerce')


#('Contains NA value:{}'.format(data.isnull().any().any()))


means_stds = None
# normalize values
means_stds = data.groupby(by=['class_name', 'bbch', 'bbch_group', 'sensor', 'crop'])['value'].agg(['mean', 'std']).reset_index()
data = data.merge(means_stds, on=['class_name', 'bbch', 'bbch_group', 'sensor', 'crop'])
means_stds = None


# standardize value
#data['value_standardized'] = (data['value'] - data['mean']) / data['std']
#data = data.drop(columns=['value', 'mean', 'std'])

# columns
#'aoi', 'bbch', 'sensor', 'fid', 'crop', 'bbch_group', 'damage', 'pattern', 'inner_x', 'inner_y',

#dt = data.pivot_table(index=['class_name', 'bbch', 'bbch_group', 'sensor', 'fid', 'crop', 'uav_date', 'aoi'], columns=['feature'], values=['value_standardized'], aggfunc=np.nanmedian)
dt = data.pivot_table(index=['class_name', 'bbch', 'bbch_group', 'sensor', 'fid', 'crop', 'uav_date', 'aoi'], columns=['feature'], values=['value'], aggfunc=np.nanmedian)

dt = dt.reset_index(col_level=1)
dt.columns = dt.columns.droplevel(0)


#data = dt.drop(columns=['class_name', 'bbch', 'bbch_group', 'sensor', 'fid', 'crop', 'uav_date', 'aoi'])
data = dt.drop(columns=['bbch', 'bbch_group', 'sensor', 'fid', 'crop', 'uav_date', 'aoi'])


# define data arrays
y = data.values[:, :1] # [:, -1] -> to get the last column of 2d array
X = data.values[:, 1:] # [:, :-1] -> to get all columns except the last one

# summarize the dataset
#st.write('shape of the y data: {}'.format(y.shape))
#st.write('shape of X data: {}'.format(X.shape))



# reshape 2d array to 1d array
y = y.ravel()
print('original labels: {}'.format(y))

ordinal_decode = False
if ordinal_decode == True:
    # decoder
    oe = OrdinalEncoder()
    oe.fit(y)
    oe.transform(y)

label_decode = True
if label_decode == True:
    #encoder
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    #to inverse: list(le.inverse_transform([2, 2, 1])


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)
X = imputer.transform(X)


# see sources =>  https://scikit-learn.org/stable/modules/preprocessing.html 6.3.1.2. Scaling sparse data
# scale the data => standardize
standardize = False
if standardize == True:
    scaler = StandardScaler(with_mean=False, with_std=True).fit(X) # recomended with_mean=False for sparse data
    print(scaler.mean_)
    print(scaler.scale_)

    X_scaled = scaler.transform(X)
    print(X_scaled)
    print(X_scaled.mean(axis=0))
    print(X_scaled.std(axis=0))


min_max_normalize = True
if min_max_normalize == True:
    min_max_scaler = MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    print(X_scaled)
    print(min_max_scaler.scale_)
    print(min_max_scaler.min_)


normalize = False
if normalize == True:
    max_abs_scaler = MaxAbsScaler()
    X_scaled = max_abs_scaler.fit_transform(X)


X = X_scaled


# extract existing features & get rid of column name 'class_name' -> labels
feature_list = data.columns.tolist()[1:]



dtree = DecisionTreeClassifier() # non-parametric classifier
rfm = RandomForestClassifier() #random_stat=101, # non-parametric classifier
xbst = XGBClassifier()
knn = KNeighborsClassifier() # non-parametric classifier
sgd = SGDClassifier()




def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


models = [dtree, rfm, xbst, knn, sgd] #clf, km, gmm, sc,svm, km, gmm, sc


model_summary = {}

#st.markdown("feature importance after different Classifiers!")
c1, c2, c3, c4, c5, = st.columns((4, 4, 4, 4, 4))
streamlit_columns = [c1, c2, c3, c4, c5]


insert_data = pd.DataFrame()

# fit the model
#yP = [None] * (len(models) + 3)
for i, model in enumerate(models):

    try:
        model.fit(X, y)
        #yP[i] = model.predict(X)

        #st.write(f'{model}'.split('(')[0])
        model_name = f'{model}'.split('(')[0]
        #st.write(model_name)

        # get importance
        if (str(model_name) == 'RandomForestClassifier') or (model_name == 'DecisionTreeClassifier') or (str(model_name) == 'XGBClassifier'):
            importance = model.feature_importances_
        elif (str(model_name) == 'SGDClassifier'):
            importance = model.coef_[0]
        elif (str(model_name) == 'KNeighborsClassifier'):

            # perform permutation importance
            result = permutation_importance(model, X, y, n_repeats=10, scoring='accuracy')
            importance = result.importances_mean
            #importance = model.coef_
        else:
            importance = None
            print('feature importance cannot be calculated !')

        if importance.any() != None:
            feature_dict = {}

            # summarize feature importance
            for ind, v in enumerate(importance):
                feature_dict[feature_list[ind]] = v
                print('Feature: %0d, Score: %.5f' % (ind, v))

            #-------- create a dict of feature as keys and importance as value
            importance_dict = dict(sorted(feature_dict.items(), key=lambda kv: kv[1]))
            df, dff = None, None

            # convert dictionary to df and sort ascending by score
            df = pd.DataFrame(list(importance_dict.items()), columns=['feature', 'score'])
            df['score_abs'] = df['score'].apply(lambda row: abs(row))
            dff = df.sort_values(by=['score_abs'], ascending=False).reset_index(drop=True).reset_index().rename(columns={'index': 'importance_rank'})

            model_summary[model_name] = importance_dict

            dff = dff.assign(crop_type_code=CROP_TYPE)
            dff = dff.assign(bbch_group=BBCH_GROUP)
            dff = dff.assign(jmd_treshold=round(JMD, 2))
            dff = dff.assign(sample_limit=SAMPLE_LIMIT)
            dff = dff.assign(top_features=TOP_FEATURES)
            dff = dff.assign(refpanel=R_Panel)
            dff = dff.assign(model=model_name)

            insert_data = pd.concat([insert_data, dff], ignore_index=True)
    except:
        print('{} did not worked'.format(model_name))

    with streamlit_columns[i]:
        st.markdown("{}".format(model_name))
        #st.dataframe(dff[['feature', 'score']].style.apply(highlight_max, subset=['score']))
        st.dataframe(dff[['feature', 'score']])



#st.write(insert_data)
with col06:
    if st.button("Calculate feature Importance & Import to DB"):

        # this needed to reopen the db in each update in the same thread
        db = dbu.connect2db(cfg.db_path)

        # ------- Insert Results
        pim_key = db.get_primary_keys('uavfimportance')
        db.insert(table='uavfimportance', primary_key=pim_key,
                  orderly_data=insert_data.to_dict(orient='records'), verbose=True, update=True)
        print('Inserted feature importance: {}'.format(model_name))




# Print default hyperparameters
#st.write("Decision Tree Classifier default hyperparameters:")
#st.write(dtree.get_params())

#st.write("\nRandom Forest Classifier default hyperparameters:")
#st.write(rfm.get_params())

#st.write("\nXGBoost Classifier default hyperparameters:")
#st.write(xbst.get_params())

#st.write("\nK-Nearest Neighbors Classifier default hyperparameters:")
#st.write(knn.get_params())

#st.write("\nStochastic Gradient Descent Classifier default hyperparameters:")
#st.write(sgd.get_params())





