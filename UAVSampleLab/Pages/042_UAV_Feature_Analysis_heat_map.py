'''
Visualisation of Feature Analysis - Jeffries Matusita Distance (JMD) Metric, queried from DB
'''

import streamlit as st
import altair as alt


import colors
import config as cfg
import dashboard_utils as dutils
from dbflow.src import db_utility as dbu




alt.data_transformers.enable('default', max_rows=None)

col01, col02 = st.columns((1, 1))
with col01:
    CROP_TYPE = st.selectbox(label='Select AOI:', options=['WW', 'SG', 'WG', 'WR', 'ZwFr', 'KM']) #, 'WG', 'WR', 'ZwFr', 'KM'
with col02:
    JMD = st.slider(label='JMD threshold:', min_value=1.0, max_value=2.0, value=1.9, step=0.1)

st.write('')



if CROP_TYPE == 'KM':
    R_Panel = 'noRP'
else:
    R_Panel = 'RP'



sql_filter = f"""
                SELECT * FROM (
                            SELECT
                                feature, 
                                jmd, 
                                class1, class2, 
                                refpanel
                                --COUNT(jmd) as count_jmd
                            FROM
                                uavfeaturejmd
                            WHERE 
                                jmd >= {JMD}
                            AND (class1  LIKE '{CROP_TYPE}%' OR class2 LIKE 'UNK%' AND class1 LIKE 'UNK%' OR class2 LIKE '{CROP_TYPE}%' )
                            AND 
                                (feature NOT LIKE '%DSM' OR feature NOT LIKE '%DSM%')
                                AND feature NOT LIKE 'GLCM%' AND feature NOT LIKE 'GLDV%' AND feature NOT LIKE 'Radius%'
                                AND feature NOT LIKE 'Elliptic%' AND feature NOT LIKE 'Roundness%' AND feature NOT LIKE 'Density%'
                                AND refpanel == '{R_Panel}'
                            GROUP BY
                                feature,
                                class1, class2
                            ORDER BY 
                                --COUNT(jmd) DESC,
                                feature 
                                ) 
                --WHERE count_jmd >= 17
                ;
            """

#AND refpanel == '{R_Panel}' # this was deleted only for Corn

number_of_samples = {
                    'WW': 89,
                    'SG': 32,
                    'WR': 26,
                    'KM': 10
                    }


if CROP_TYPE == 'KM':
    Proportion_for_Feature_Importance = 0.20
elif CROP_TYPE == 'WR':
    Proportion_for_Feature_Importance = 0.15
elif CROP_TYPE == 'WW':
    Proportion_for_Feature_Importance = 0.15
else:
    Proportion_for_Feature_Importance = 0.15



# Calculate SELECT_THRESH

#st.write(SELECT_THRESH)

number_of_samples = {
                    'WW': 66 - 20,
                    'WR': 60 - 20,
                    'SG': 27 - 20,
                    'KM': 18 - 18
                    }

SELECT_THRESH = number_of_samples[CROP_TYPE]
SELECT_THRESH=0


#SELECT_THRESH = 18 # only fo Corn
df = dbu.query_sql(sql_filter, db_engine=cfg.dbarchive.archive.engine)

highlight = alt.selection_multi(fields=['count'], bind='legend')
click = alt.selection_multi(encodings=['x', 'y'], bind='legend')

df2 = dutils.feature_layer_split(df)['df_main']
df1 = dutils.feature_layer_split(df)['df_texture']
source = dutils.feature_layer_split(df)['df']

# Step 1: Aggregate and filter the data in Pandas
aggregated_data = df2.groupby(['layers', 'feature', 'feature_level']).size().reset_index(name='count')
filtered_data = aggregated_data[aggregated_data['count'] > SELECT_THRESH]
unique_features = filtered_data['feature'].unique()

st.write(df2)


df2['features'] = df2['features'].replace('quantile[50]', 'Median')


# Create a custom color scale transitioning from green to blue
custome_virdis = [

               '#482878',
               '#3E4989',
               '#31688E',
               '#26828E',
               '#1F9E89',
               '#35B779',
               '#6DCC5B',
               #'#B4DD2C',
               #'#FDE725'
            ]
custom_scale = alt.Scale(
    range=custome_virdis[::-1]  # Green to blue
)

df2 = df2.query("layers != 'DSM'"
                "& features != 'mode[Median]' "
                "& layers != 'index' "
                )

#df2 = df2.drop('feature_level', axis=1)
st.write(df2)

heatmap1 = alt.Chart(df2
).transform_aggregate(
    count='count():Q',
    groupby=['layers', 'features', 'feature_level']
    #groupby=['layers', 'features']
).transform_filter(
    alt.datum.count > SELECT_THRESH  # Filtering the counts to be greater than 60
).mark_rect(stroke='lightgrey', strokeWidth=0.1
).encode(
    x=alt.X('layers:O',
            sort=alt.EncodingSortField(field='count', op='max', order='descending'),
            axis=alt.Axis(labelFontSize=28, labelColor='black', labelLimit=300, titleColor='black', titleFontSize=30,
                          labelAlign='right', labelPadding=10),
            title=''
            #title = 'Layers of {}'.format(colors.crop_types[CROP_TYPE])
            ),
    y=alt.Y('features:O',
            sort=alt.EncodingSortField(field='count', op='max', order='descending'),
            axis=alt.Axis(labelFontSize=28, labelColor='black', labelLimit=300, titleColor='black', titleFontSize=30,
                          labelAlign='right', labelPadding=10),
            title=''
            ),
    color=alt.Color('count:Q',
                    scale=custom_scale,
                    #scale=alt.Scale(scheme='viridis'),
                    legend=alt.Legend(titleFontSize=32, titleColor='black', labelFontSize=30, title='Count', labelColor='black')), # scheme='redblue',

    opacity=alt.condition(highlight, alt.value(1), alt.value(0.2)),
    tooltip=['features', 'layers', 'count:Q', 'feature_level:O'],
).properties(
    width=1650,
    height=470,
    #width=1650, #/ 1.1, # only for Corn
    #height=470/ 1.23, # only for Corn
    title=alt.TitleParams(text='{}'.format(colors.crop_types[CROP_TYPE]), fontSize=22, align='center')
#).configure_legend(titleColor='black', titleFontSize=14
).add_selection(
    highlight,
    #click,
)


heatmap1.configure(
    font='Helvetica',
    axis=alt.AxisConfig(labelFont='Helvetica', titleFont='Helvetica'),
    legend=alt.LegendConfig(labelFont='Helvetica', titleFont='Helvetica'),
    title=alt.TitleConfig(font='Helvetica')
)
st.write(heatmap1)



# Remove the first three characters (i.e., 'SG_') from 'class1' and 'class2' columns
df2['class1'] = df2['class1'].str.slice(3)
df2['class2'] = df2['class2'].str.slice(3)


# Create the heatmap
heatmap_texture = alt.Chart(df2).transform_aggregate(
    count='count(jmd):Q',
    groupby=['class1', 'class2']
).encode(
    alt.X('class1:O', title=''),
    alt.Y('class2:O', title=''),
    opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
    tooltip=['class1', 'class2', 'count:Q']
).mark_rect(stroke='lightgrey', strokeWidth=1).encode(
    color=alt.Color('count:Q', scale=alt.Scale(scheme='greenblue')
                    , legend=alt.Legend(title='Count')
                    )
).add_selection(
    highlight,
 #   selector,
    click
)

heatmap_texture.configure(
    font='Helvetica',
    axis=alt.AxisConfig(labelFont='Helvetica', titleFont='Helvetica'),
    legend=alt.LegendConfig(labelFont='Helvetica', titleFont='Helvetica'),
    title=alt.TitleConfig(font='Helvetica')
)

st.write(heatmap_texture)



heatmaps = alt.hconcat(heatmap1,
                     heatmap_texture
                     )#.resolve_scale(
                     #x='shared',
                     #y='shared'
                     #)



# Extract the transformed data from the chart
#chart_dict = heatmap1.to_dict()

# Find the transformed data in the chart's specification
#transformed_data = chart_dict['datasets'][list(chart_dict['datasets'].keys())[0]]

# Convert the transformed data to a Pandas DataFrame
#transformed_df = pd.DataFrame(transformed_data)



# Step 1: Extract unique values from 'features' column
#unique_features = transformed_df['feature'].unique()


# Step 2: Write the unique values to a text file
with open(r'{}\_temp\unique_features_{}.txt'.format(cfg.output_path, CROP_TYPE), 'w') as f:
    for feature in unique_features:
        f.write(f"{feature}\n")



col21, col22 = st.columns((1, 5))
#with col21:
#    JMD = st.slider(label='JMD threshold:', min_value=1.0, max_value=2.0, value=1.9, step=0.1)


#slider = alt.binding_range(min=0, max=15, step=1, name='cutoff:')
#selector = alt.selection_single(name="SelectorName", fields=['count'],
#                                bind=slider, init={'count(jmd):Q': 3})


# Configure common options
rect = alt.Chart(source
).transform_aggregate(
    count='count():O',
    groupby=['class1', 'class2']
).encode(
    alt.X('class1:O', scale=alt.Scale(paddingInner=0)),
    alt.Y('class2:O', scale=alt.Scale(paddingInner=0)),
    #color=alt.condition(highlight, 'count()', alt.value('lightgray')),
    opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
    tooltip=['class1', 'class2', 'count()']
).mark_rect(stroke='lightgrey', strokeWidth=1
).encode(
        color=alt.Color('count(jmd):O', scale=alt.Scale(scheme='greenblue')),
        #order=alt.Order('count(jmd):O', sort='ascending')
        #legend=alt.Legend(direction='horizontal'))
).add_selection(
    highlight,
 #   selector,
    click
)



# ---- PLot Heat plots
heatmap3 = alt.Chart(source
).mark_rect(stroke='lightgrey', strokeWidth=1).encode(
    x=alt.X('layers:O', sort=alt.EncodingSortField(field='layers', op='count', order='descending'), title='layers/degree'),
    y=alt.Y('features:O', sort=alt.EncodingSortField(field='features', op='count', order='ascending')),
    color=alt.Color('count(jmd):Q', scale=alt.Scale(scheme='greenblue')),
    tooltip=['features', 'layers',  'count(jmd)', 'feature_level:N']
).transform_filter(
    click,
).properties(
    width=600,
    height=450
).add_selection(
    highlight
)


plot = alt.hconcat(
    heatmap3,
    rect,
)


plot_text = alt.vconcat(plot)
st.write(plot_text)

